package ann

import (
	"github.com/marcospedreiro/tann/tensor"
)

// Ann interface for an artificial neural network
type Ann interface {
	Train(*tensor.Tensor, *tensor.Tensor, int) error

	Think(*tensor.Tensor) (*tensor.Tensor, error)
}
