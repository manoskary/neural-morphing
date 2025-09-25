#pragma once

#include <mutex>
#include <vector>

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

class PaletteIndex
{
public:
    explicit PaletteIndex(int dimensions);

    void clear();
    void add(const std::vector<float>& vectorRow, PaletteMeta meta);
    void build();

    std::vector<MatchResult> query(const std::vector<float>& queryVector, int k) const;

    int dimensions() const noexcept { return dims_; }
    const PaletteMeta& meta(int index) const { return metas_.at(index); }
    int size() const noexcept { return static_cast<int>(vectors_.size()); }

private:
    float cosineDistance(const std::vector<float>& a, const std::vector<float>& b) const;

    int dims_ = 0;
    std::vector<std::vector<float>> vectors_;
    std::vector<PaletteMeta> metas_;
    std::vector<float> norms_;
    mutable std::mutex mutex_;
};
