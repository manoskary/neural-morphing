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

void PaletteIndex::build()
{
    // Placeholder: when switching to ANN, build the index here.
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
        const float dist = cosineDistance(queryVector, vectors_[i]) / (queryNorm * norms_[i] + 1.0e-9f);
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

float PaletteIndex::cosineDistance(const std::vector<float>& a, const std::vector<float>& b) const
{
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        dot += a[i] * b[i];

    return 1.0f - dot;
}
