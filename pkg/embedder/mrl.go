package embedder

import (
	"fmt"
)

// Supported standard MRL (Matryoshka Representation Learning) dimensions.
var SupportedMRLDims = []int{512, 1024, 1536, 3024}

// MRLSlicer provides slicing and unit re-normalization for nested subspace representations.
type MRLSlicer struct{}

// NewMRLSlicer creates a new MRLSlicer.
func NewMRLSlicer() *MRLSlicer {
	return &MRLSlicer{}
}

// Slice extracts the prefix sub-vector of dimension `dim` from `vec` and unit-normalizes it.
func (s *MRLSlicer) Slice(vec []float32, dim int) ([]float32, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("target dimension must be positive, got %d", dim)
	}
	if len(vec) < dim {
		return nil, fmt.Errorf("vector length %d is smaller than requested dimension %d", len(vec), dim)
	}

	sliced := make([]float32, dim)
	copy(sliced, vec[:dim])
	return L2Normalize(sliced), nil
}

// SliceAll extracts all standard MRL dimensions (512, 1024, 1536, 3024) with unit normalization.
func (s *MRLSlicer) SliceAll(vec []float32) (map[int][]float32, error) {
	if len(vec) != TotalDim {
		return nil, fmt.Errorf("full vector must have %d dimensions, got %d", TotalDim, len(vec))
	}

	slices := make(map[int][]float32, len(SupportedMRLDims))
	for _, dim := range SupportedMRLDims {
		sub, err := s.Slice(vec, dim)
		if err != nil {
			return nil, err
		}
		slices[dim] = sub
	}
	return slices, nil
}

// CompareMRL calculates cosine similarities between two 3024-d vectors across all standard MRL dimensions.
func (s *MRLSlicer) CompareMRL(v1, v2 []float32) (map[int]float32, error) {
	if len(v1) != TotalDim || len(v2) != TotalDim {
		return nil, fmt.Errorf("both vectors must have %d dimensions", TotalDim)
	}

	results := make(map[int]float32, len(SupportedMRLDims))
	for _, dim := range SupportedMRLDims {
		sub1, err := s.Slice(v1, dim)
		if err != nil {
			return nil, err
		}
		sub2, err := s.Slice(v2, dim)
		if err != nil {
			return nil, err
		}
		results[dim] = CosineSimilarity(sub1, sub2)
	}
	return results, nil
}
