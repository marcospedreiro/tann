package tensor

import (
	"fmt"
	"math"
)

// DotProduct of two 1xn vectors = a1*b1 + a2*b2 + ... + an*bn
func DotProduct(a []float64, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector length mismatch: len of a %d, len of b %d", len(a), len(b))
	}
	dotProduct := 0.0
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
	}
	return dotProduct, nil
}

// AddScalar adds to every element in the tensor
func (t *Tensor) AddScalar(x float64) {
	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = t.Data[i][j] + x
		}
	}
}

// MultiplyScalar multiplies every element in the tensor
func (t *Tensor) MultiplyScalar(x float64) {
	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = t.Data[i][j] * x
		}
	}
}

// SubtractScalar subtracts from every element in the tensor (element - x)
func (t *Tensor) SubtractScalar(x float64) {
	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = t.Data[i][j] - x
		}
	}
}

// ScalarSubtractByTensor subtracts x by every element in the tensor (x - element)
func (t *Tensor) ScalarSubtractByTensor(x float64) {
	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = x - t.Data[i][j]
		}
	}
}

// DivideScalar divides every element in the tensor (element / x)
func (t *Tensor) DivideScalar(x float64) {
	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = t.Data[i][j] / x
		}
	}
}

// ScalarDivideByTensor divides x by every element in the tensor (x / element)
func (t *Tensor) ScalarDivideByTensor(x float64) {
	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = x / t.Data[i][j]
		}
	}
}

// NaturalExp performs e^x where x is each element of the tensor
func (t *Tensor) NaturalExp() {
	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = math.Exp(t.Data[i][j])
		}
	}
}

// AddTensor performs matrix addition and computes t + b where t and b are two matrices
func (t *Tensor) AddTensor(b *Tensor) error {
	err := t.Validate()
	if err != nil {
		return fmt.Errorf("current tensor is not valid: %s", err.Error())
	}
	err = b.Validate()
	if err != nil {
		return fmt.Errorf("provided tensor is not valid: %s", err.Error())
	}

	if t.Rows != b.Rows {
		return fmt.Errorf("tensor dimension mismatch: has %d, got %d", t.Rows, b.Rows)
	}

	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {

		if len(t.Data[i]) != len(b.Data[i]) {
			return fmt.Errorf("dimension length mismatch: has %d, got %d", len(t.Data[i]), len(b.Data[i]))
		}

		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = t.Data[i][j] + b.Data[i][j]
		}
	}

	return nil
}

// SubtractTensor performs matrix addition and computes t - b where t and b are two matrices
func (t *Tensor) SubtractTensor(b *Tensor) error {
	err := t.Validate()
	if err != nil {
		return fmt.Errorf("current tensor is not valid: %s", err.Error())
	}
	err = b.Validate()
	if err != nil {
		return fmt.Errorf("provided tensor is not valid: %s", err.Error())
	}

	if t.Rows != b.Rows {
		return fmt.Errorf("tensor dimension mismatch: has %d, got %d", t.Rows, b.Rows)
	}

	dimLen := len(t.Data[0])
	for i := 0; i < t.Rows; i++ {

		if len(t.Data[i]) != len(b.Data[i]) {
			return fmt.Errorf("dimension length mismatch: has %d, got %d", len(t.Data[i]), len(b.Data[i]))
		}

		for j := 0; j < dimLen; j++ {
			t.Data[i][j] = t.Data[i][j] - b.Data[i][j]
		}
	}

	return nil
}

/*MultiplyTensor if t is an NxM matrix, and b is an MxP matrix, their matrix
product t*b will be an NxP matrix. Its computational complexity is O(n^3) for nxn
matrices for the basic algorithm (this complexity is O(n^2.373) for the
asymptotically fastest known algorithm.
*/
func (t *Tensor) MultiplyTensor(b *Tensor) (*Tensor, error) {
	err := t.Validate()
	if err != nil {
		return nil, fmt.Errorf("current tensor is not valid: %s", err.Error())
	}
	err = b.Validate()
	if err != nil {
		return nil, fmt.Errorf("provided tensor is not valid: %s", err.Error())
	}

	// cols in t must match rows in b
	if t.Columns != b.Rows {
		return nil, fmt.Errorf("t columns (%d) not equal to b rows (%d)", t.Columns, b.Rows)
	}

	// make new tensor to hold t*b
	productMatrix := New(t.Rows, b.Columns)

	for i := 0; i < productMatrix.Rows; i++ {
		for j := 0; j < productMatrix.Columns; j++ {
			productMatrix.Data[i][j], err = DotProduct(t.Data[i], b.Data[:][j])
			if err != nil {
				return nil, fmt.Errorf("failure calculating dot product during matrix multiplication: %s", err.Error())
			}
		}
	}

	return productMatrix, nil
}
