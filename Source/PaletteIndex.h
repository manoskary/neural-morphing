#pragma once

#include <mutex>
#include <vector>

#include "ModelBackend.h"

struct PaletteMeta
{
    int sampleId = -1;
    int frame = -1;
};

struct MatchResult
{
    int annIndex = -1;
    float distance = 0.0f;
};

struct GrainConfig
{
    int unit = 1;
    int stride = 1;
};

class PaletteIndex
{
public:
    explicit PaletteIndex(int dimensions);

    void clear();
    void add(const std::vector<float>& vectorRow, PaletteMeta meta);
    void build();

    std::vector<MatchResult> query(const std::vector<float>& queryVector, int k) const;

    void setGrainConfig(int unit, int stride);
    GrainConfig grainConfig() const;
    bool getVector(int index, std::vector<float>& out) const;
    float cosineDistance(int indexA, int indexB) const;

    void prepareForSamples(int count);
    void setTokenBlock(int sampleId, TokenBlock block);
    const TokenBlock* tokenBlockForSample(int sampleId) const;
    const TokenBlock* tokensForMeta(const PaletteMeta& meta) const;

    int dimensions() const noexcept { return dims_; }
    const PaletteMeta& meta(int index) const { return metas_.at(index); }
    int size() const noexcept { return static_cast<int>(vectors_.size()); }

private:
    int dims_ = 0;
    std::vector<std::vector<float>> vectors_;
    std::vector<PaletteMeta> metas_;
    std::vector<float> norms_;
    std::vector<TokenBlock> tokenBlocks_;
    GrainConfig grainConfig_;
    mutable std::mutex mutex_;
};
