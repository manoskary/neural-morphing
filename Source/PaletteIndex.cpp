#include "PaletteIndex.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace
{
float cosineDistanceSlice(const std::vector<float>& query,
                          const float* candidate,
                          int start,
                          int length)
{
    if (length <= 0)
        return 0.0f;

    float dot = 0.0f;
    float queryNorm = 0.0f;
    float candidateNorm = 0.0f;
    for (int i = 0; i < length; ++i)
    {
        const float q = query[static_cast<size_t>(start + i)];
        const float c = candidate[start + i];
        dot += q * c;
        queryNorm += q * q;
        candidateNorm += c * c;
    }

    return 1.0f - dot / (std::sqrt(queryNorm * candidateNorm) + 1.0e-9f);
}

float cosineDistanceRaw(const float* first, const float* second, int length)
{
    float dot = 0.0f;
    float firstNorm = 0.0f;
    float secondNorm = 0.0f;
    for (int i = 0; i < length; ++i)
    {
        dot += first[i] * second[i];
        firstNorm += first[i] * first[i];
        secondNorm += second[i] * second[i];
    }
    return 1.0f - dot / (std::sqrt(firstNorm * secondNorm) + 1.0e-9f);
}
}

PaletteIndex::PaletteIndex(int dimensions)
    : dims_(dimensions)
{
    auto empty = std::make_shared<Snapshot>();
    empty->dimensions = dims_;
    std::atomic_store(&snapshot_, SnapshotPtr(empty));
}

void PaletteIndex::clear()
{
    {
        std::scoped_lock lock(pendingMutex_);
        pendingSources_.reset();
    }

    auto empty = std::make_shared<Snapshot>();
    empty->dimensions = dims_;
    std::atomic_store(&snapshot_, SnapshotPtr(empty));
}

void PaletteIndex::prepareForSamples(int count)
{
    auto sources = std::make_shared<SourceData>();
    const auto size = static_cast<size_t>(std::max(0, count));
    sources->tokenBlocks.resize(size);
    sources->frameVectors.resize(size);

    std::scoped_lock lock(pendingMutex_);
    pendingSources_ = std::move(sources);
}

void PaletteIndex::setSampleData(int sampleId,
                                 TokenBlock block,
                                 std::vector<std::vector<float>> frameVectors)
{
    if (sampleId < 0)
        return;

    std::scoped_lock lock(pendingMutex_);
    if (pendingSources_ == nullptr)
        return;

    const auto index = static_cast<size_t>(sampleId);
    if (index >= pendingSources_->tokenBlocks.size())
    {
        pendingSources_->tokenBlocks.resize(index + 1);
        pendingSources_->frameVectors.resize(index + 1);
    }

    pendingSources_->tokenBlocks[index] = std::move(block);
    pendingSources_->frameVectors[index] = std::move(frameVectors);
}

PaletteIndex::SnapshotPtr PaletteIndex::buildSnapshot(std::shared_ptr<const SourceData> sources,
                                                       int unit,
                                                       int stride) const
{
    auto result = std::make_shared<Snapshot>();
    result->sources = std::move(sources);
    result->dimensions = dims_;
    result->grainConfig.unit = std::max(1, unit);
    result->grainConfig.stride = std::min(std::max(1, stride), result->grainConfig.unit);

    if (result->sources == nullptr)
        return result;

    for (size_t sampleId = 0; sampleId < result->sources->tokenBlocks.size(); ++sampleId)
    {
        const auto& tokens = result->sources->tokenBlocks[sampleId];
        const auto& frames = result->sources->frameVectors[sampleId];
        const int frameCount = std::min(tokens.frames, static_cast<int>(frames.size()));
        if (tokens.empty() || frameCount < result->grainConfig.unit)
            continue;

        for (int frame = 0;
             frame + result->grainConfig.unit <= frameCount;
             frame += result->grainConfig.stride)
        {
            const auto vectorStart = result->vectors.size();
            result->vectors.resize(vectorStart + static_cast<size_t>(dims_), 0.0f);
            bool valid = true;

            for (int offset = 0; offset < result->grainConfig.unit; ++offset)
            {
                const auto& row = frames[static_cast<size_t>(frame + offset)];
                if (static_cast<int>(row.size()) != dims_)
                {
                    valid = false;
                    break;
                }

                for (int dimension = 0; dimension < dims_; ++dimension)
                    result->vectors[vectorStart + static_cast<size_t>(dimension)] += row[static_cast<size_t>(dimension)];
            }

            if (!valid)
            {
                result->vectors.resize(vectorStart);
                continue;
            }

            const float scale = 1.0f / static_cast<float>(result->grainConfig.unit);
            for (int dimension = 0; dimension < dims_; ++dimension)
                result->vectors[vectorStart + static_cast<size_t>(dimension)] *= scale;

            result->metas.push_back({ static_cast<int>(sampleId), frame });
        }
    }

    return result;
}

bool PaletteIndex::publishBuild(int unit, int stride)
{
    std::shared_ptr<SourceData> sources;
    {
        std::scoped_lock lock(pendingMutex_);
        sources = std::move(pendingSources_);
    }

    if (sources == nullptr)
        return false;

    auto next = buildSnapshot(std::move(sources), unit, stride);
    const bool ready = next->size() > 0;
    if (ready)
        std::atomic_store(&snapshot_, SnapshotPtr(std::move(next)));
    return ready;
}

bool PaletteIndex::rebuildGrains(int unit, int stride)
{
    const auto current = snapshot();
    if (current == nullptr || current->sources == nullptr)
        return false;

    auto next = buildSnapshot(current->sources, unit, stride);
    const bool ready = next->size() > 0;
    if (ready)
        std::atomic_store(&snapshot_, SnapshotPtr(std::move(next)));
    return ready;
}

PaletteIndex::SnapshotPtr PaletteIndex::snapshot() const
{
    return std::atomic_load(&snapshot_);
}

std::vector<MatchResult> PaletteIndex::query(const SnapshotPtr& data,
                                             const std::vector<float>& queryVector,
                                             int k,
                                             const RvqSearchConfig& search) const
{
    std::vector<MatchResult> results;
    if (data == nullptr || k <= 0 || static_cast<int>(queryVector.size()) != dims_)
        return results;

    const int count = data->size();
    results.reserve(static_cast<size_t>(count));

    const bool grouped = search.coarseLength + search.midLength + search.fineLength == dims_;
    for (int index = 0; index < count; ++index)
    {
        const float* candidate = data->vectors.data() + static_cast<size_t>(index) * static_cast<size_t>(dims_);
        MatchResult match;
        match.annIndex = index;

        if (grouped)
        {
            match.coarseDistance = cosineDistanceSlice(queryVector, candidate, 0, search.coarseLength);
            match.midDistance = cosineDistanceSlice(queryVector, candidate, search.coarseLength, search.midLength);
            match.fineDistance = cosineDistanceSlice(
                queryVector,
                candidate,
                search.coarseLength + search.midLength,
                search.fineLength);
            match.distance = search.coarseWeight * match.coarseDistance
                             + search.midWeight * match.midDistance
                             + search.fineWeight * match.fineDistance;
        }
        else
        {
            match.distance = cosineDistanceSlice(queryVector, candidate, 0, dims_);
            match.coarseDistance = match.distance;
            match.fineDistance = match.distance;
        }
        results.push_back(match);
    }

    const auto keep = std::min(results.size(), static_cast<size_t>(k));
    std::partial_sort(results.begin(), results.begin() + static_cast<std::ptrdiff_t>(keep), results.end(),
                      [](const MatchResult& lhs, const MatchResult& rhs)
                      {
                          if (lhs.distance == rhs.distance)
                              return lhs.annIndex < rhs.annIndex;
                          return lhs.distance < rhs.distance;
                      });
    results.resize(keep);
    return results;
}

float PaletteIndex::cosineDistance(const SnapshotPtr& data, int indexA, int indexB) const
{
    if (data == nullptr || indexA < 0 || indexB < 0 || indexA >= data->size() || indexB >= data->size())
        return 1.0f;

    const float* firstVector = data->vectors.data() + static_cast<size_t>(indexA) * static_cast<size_t>(dims_);
    const float* secondVector = data->vectors.data() + static_cast<size_t>(indexB) * static_cast<size_t>(dims_);
    return cosineDistanceRaw(firstVector, secondVector, dims_);
}

GrainConfig PaletteIndex::grainConfig() const
{
    const auto current = snapshot();
    return current != nullptr ? current->grainConfig : GrainConfig{};
}

int PaletteIndex::size() const noexcept
{
    const auto current = snapshot();
    return current != nullptr ? current->size() : 0;
}
