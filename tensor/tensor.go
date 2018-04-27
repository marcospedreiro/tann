package tensor

import "fmt"

/* invariants:

[][]float64

first array is for each dimension

second array is for data in that dimension
All secondary arrays must be of the same length at all times
*/

// Tensor represents an n dimensional array
type Tensor struct {
	Data [][]float64

	Rows    int
	Columns int
}

// New returns a pointer to a tensor of dimensions DxN
func New(d int, n int) *Tensor {
	newTensor := &Tensor{
		Rows:    d,
		Columns: n,
	}

	for i := 0; i < d; i++ {
		newTensor.Data = append(newTensor.Data, make([]float64, n))
	}

	return newTensor
}

// Validate whether or not a tensor is valid. O(N) where N is the number of rows in the Tensor
func (t *Tensor) Validate() error {
	/* len() is constant time on arrays, slices, and maps
	   This method is O(N) where N is the number of dimensions in the Tensor
	*/
	if len(t.Data) != t.Rows {
		return fmt.Errorf("row number mismatch: expected %d rows, found %d", t.Rows, len(t.Data))
	}
	if t.Rows > 0 {
		// make sure each row is of same length
		for i := 0; i < t.Rows; i++ {
			if len(t.Data[i]) != t.Columns {
				return fmt.Errorf("column length mismatch: expected column length %d, found %d", t.Columns, len(t.Data[i]))
			}
		}
	}
	return nil
}

// Append new data to a tensor, new data must have the same rows as the tensor
func (t *Tensor) Append(TensorToAppend *Tensor) error {
	err := t.Validate()
	if err != nil {
		return fmt.Errorf("current tensor is not valid: %s", err.Error())
	}
	err = TensorToAppend.Validate()
	if err != nil {
		return fmt.Errorf("provided tensor is not valid: %s", err.Error())
	}
	if t.Rows != TensorToAppend.Rows {
		return fmt.Errorf("tensor row mismatch: has %d rows, got tensor with %d rows", t.Rows, TensorToAppend.Rows)
	}

	for i := 0; i < t.Rows; i++ {
		t.Data[i] = append(t.Data[i], TensorToAppend.Data[i]...)
	}
	t.Columns = len(t.Data[0])
	return nil
}

// Transpose the tensor MxN -> NxM as a new tensor
func (t *Tensor) Transpose() *Tensor {
	transpose := New(t.Columns, t.Rows)

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Columns; j++ {
			transpose.Data[j][i] = t.Data[i][j]
		}
	}

	return transpose
}

// GetColumn c of tensor
func (t *Tensor) GetColumn(c int) ([]float64, error) {
	if c < 0 || c >= t.Columns {
		return nil, fmt.Errorf("c (%d) out of bounds either < 0 or >= Columns (%d)", c, t.Columns)
	}

	column := make([]float64, 0)
	for i := 0; i < t.Rows; i++ {
		column = append(column, t.Data[i][c])
	}
	return column, nil

}
