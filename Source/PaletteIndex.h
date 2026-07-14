#pragma once

#include <memory>
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
    float coarseDistance = 0.0f;
    float midDistance = 0.0f;
    float fineDistance = 0.0f;
};

struct GrainConfig
{
    int unit = 1;
    int stride = 1;
};

struct RvqSearchConfig
{
    int coarseLength = 0;
    int midLength = 0;
    int fineLength = 0;
    float coarseWeight = 1.0f;
    float midWeight = 0.0f;
    float fineWeight = 0.0f;
};

class PaletteIndex
{
public:
    struct SourceData
    {
        std::vector<TokenBlock> tokenBlocks;
        std::vector<std::vector<std::vector<float>>> frameVectors;
    };

    struct Snapshot
    {
        std::shared_ptr<const SourceData> sources;
        std::vector<float> vectors;
        std::vector<PaletteMeta> metas;
        GrainConfig grainConfig;
        int dimensions = 0;

        int size() const noexcept
        {
            return dimensions > 0 ? static_cast<int>(vectors.size() / static_cast<size_t>(dimensions)) : 0;
        }
    };

    using SnapshotPtr = std::shared_ptr<const Snapshot>;

    explicit PaletteIndex(int dimensions);

    void clear();
    void prepareForSamples(int count);
    void setSampleData(int sampleId, TokenBlock block, std::vector<std::vector<float>> frameVectors);
    bool publishBuild(int unit, int stride);
    bool rebuildGrains(int unit, int stride);

    SnapshotPtr snapshot() const;
    std::vector<MatchResult> query(const SnapshotPtr& snapshot,
                                   const std::vector<float>& queryVector,
                                   int k,
                                   const RvqSearchConfig& search) const;
    float cosineDistance(const SnapshotPtr& snapshot, int indexA, int indexB) const;

    GrainConfig grainConfig() const;
    int dimensions() const noexcept { return dims_; }
    int size() const noexcept;

private:
    SnapshotPtr buildSnapshot(std::shared_ptr<const SourceData> sources, int unit, int stride) const;

    int dims_ = 0;
    mutable std::mutex pendingMutex_;
    std::shared_ptr<SourceData> pendingSources_;
    SnapshotPtr snapshot_;
};
