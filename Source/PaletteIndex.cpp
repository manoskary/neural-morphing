#include "PaletteIndex.h"

#include <algorithm>
#include <cmath>
#include <numeric>

PaletteIndex::PaletteIndex(int dimensions)
    : dims_(dimensions)
{
}

void PaletteIndex::clear()
{
    std::scoped_lock lock(mutex_);
    vectors_.clear();
    metas_.clear();
    norms_.clear();
    tokenBlocks_.clear();
    grainConfig_ = {};
}

void PaletteIndex::add(const std::vector<float>& vectorRow, PaletteMeta meta)
{
    if (static_cast<int>(vectorRow.size()) != dims_)
        return;

    std::scoped_lock lock(mutex_);
    vectors_.push_back(vectorRow);
    metas_.push_back(meta);

    const float norm = std::sqrt(std::inner_product(vectorRow.begin(), vectorRow.end(), vectorRow.begin(), 0.0f) + 1.0e-9f);
    norms_.push_back(norm);
}

void PaletteIndex::prepareForSamples(int count)
{
    std::scoped_lock lock(mutex_);
    tokenBlocks_.clear();
    tokenBlocks_.resize(static_cast<size_t>(std::max(0, count)));
}

void PaletteIndex::setTokenBlock(int sampleId, TokenBlock block)
{
    if (sampleId < 0)
        return;

    std::scoped_lock lock(mutex_);
    if (static_cast<size_t>(sampleId) >= tokenBlocks_.size())
        tokenBlocks_.resize(static_cast<size_t>(sampleId) + 1);

    tokenBlocks_[static_cast<size_t>(sampleId)] = std::move(block);
}

const TokenBlock* PaletteIndex::tokenBlockForSample(int sampleId) const
{
    if (sampleId < 0)
        return nullptr;

    std::scoped_lock lock(mutex_);
    if (static_cast<size_t>(sampleId) >= tokenBlocks_.size())
        return nullptr;

    const auto& block = tokenBlocks_[static_cast<size_t>(sampleId)];
    if (block.tokens.empty())
        return nullptr;

    return &block;
}

const TokenBlock* PaletteIndex::tokensForMeta(const PaletteMeta& meta) const
{
    return tokenBlockForSample(meta.sampleId);
}

void PaletteIndex::build()
{
    // Placeholder: when switching to ANN, build the index here.
}

void PaletteIndex::setGrainConfig(int unit, int stride)
{
    std::scoped_lock lock(mutex_);
    grainConfig_.unit = std::max(1, unit);
    grainConfig_.stride = std::max(1, stride);
}

GrainConfig PaletteIndex::grainConfig() const
{
    std::scoped_lock lock(mutex_);
    return grainConfig_;
}

bool PaletteIndex::getVector(int index, std::vector<float>& out) const
{
    std::scoped_lock lock(mutex_);
    if (index < 0 || static_cast<size_t>(index) >= vectors_.size())
        return false;

    out = vectors_[static_cast<size_t>(index)];
    return true;
}

float PaletteIndex::cosineDistance(int indexA, int indexB) const
{
    std::scoped_lock lock(mutex_);
    if (indexA < 0 || indexB < 0)
        return 1.0f;

    const size_t idxA = static_cast<size_t>(indexA);
    const size_t idxB = static_cast<size_t>(indexB);
    if (idxA >= vectors_.size() || idxB >= vectors_.size())
        return 1.0f;

    float dot = 0.0f;
    for (size_t i = 0; i < vectors_[idxA].size(); ++i)
        dot += vectors_[idxA][i] * vectors_[idxB][i];

    const float denom = (norms_[idxA] * norms_[idxB]) + 1.0e-9f;
    return 1.0f - dot / denom;
}

std::vector<MatchResult> PaletteIndex::query(const std::vector<float>& queryVector, int k) const
{
    std::vector<MatchResult> results;
    results.reserve(static_cast<size_t>(k));

    if (queryVector.size() != static_cast<size_t>(dims_))
        return results;

    const float queryNorm = std::sqrt(std::inner_product(queryVector.begin(), queryVector.end(), queryVector.begin(), 0.0f) + 1.0e-9f);

    std::scoped_lock lock(mutex_);
    for (size_t i = 0; i < vectors_.size(); ++i)
    {
        const float dot = std::inner_product(queryVector.begin(), queryVector.end(), vectors_[i].begin(), 0.0f);
        const float dist = 1.0f - dot / (queryNorm * norms_[i] + 1.0e-9f);
        results.push_back({ static_cast<int>(i), dist });
    }

    const auto slice = std::min(results.size(), static_cast<size_t>(k));
    std::partial_sort(results.begin(), results.begin() + slice, results.end(),
        [](const MatchResult& lhs, const MatchResult& rhs)
        {
            return lhs.distance < rhs.distance;
        });

    if (static_cast<int>(results.size()) > k)
        results.resize(static_cast<size_t>(k));

    return results;
}
