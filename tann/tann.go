package tann

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/marcospedreiro/tann/tensor"
)

// Tann is a thing
type Tann struct {
	/*Seed the random number generator, so it generates the same numbers
	  every time the program runs.*/
	RandomSeed float64

	/*We model a single neuron, with 3 input connections and 1 output connection.
	  We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
	  and mean 0.*/
	SynapticWeights *tensor.Tensor
}

// New returns a new Tann of DxN with a RandomSeed and intial SynapticWeights
func New(c int) *Tann {
	rows := 1
	rseed := rand.Float64()

	t := tensor.New(rows, c)
	for i := 0; i < rows; i++ {
		for j := 0; j < c; j++ {
			t.Data[i][j] = -1 + rand.Float64()*(1 - -1)
		}
	}

	return &Tann{
		RandomSeed:      rseed,
		SynapticWeights: t,
	}
}

/* The Sigmoid function, which describes an S shaped curve.
We pass the weighted sum of the inputs through this function to normalise them between 0 and 1. */
func (tn *Tann) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

/* The derivative of the Sigmoid function. This is the gradient of the Sigmoid curve.
It indicates how confident we are about the existing weight.*/
func (tn *Tann) sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// Train the neural network on the training set for a determined number of iterations
func (tn *Tann) Train(trainingInputs *tensor.Tensor, trainingOutputs *tensor.Tensor, trainingIterations int) error {

	if trainingOutputs.Rows != trainingInputs.Rows {
		return fmt.Errorf("trainingOutputs rows (%d) do not match trainingInputs rows (%d)", trainingOutputs.Rows, trainingInputs.Rows)
	}
	if trainingInputs.Columns != tn.SynapticWeights.Columns {
		return fmt.Errorf("trainingInputs columns (%d) do not match synaptic weight columns (%d)", trainingInputs.Columns, tn.SynapticWeights.Columns)
	}

	for i := 0; i < trainingIterations; i++ {
		out, err := tn.Think(trainingInputs)
		if err != nil {
			return err
		}

		trainErr := tensor.New(trainingOutputs.Rows, trainingOutputs.Columns)
		trainErr.Data = trainingOutputs.Data

		err = trainErr.SubtractTensor(out)
		if err != nil {
			return err
		}

		trainErrMultiplied := tensor.New(trainingInputs.Rows, 0)
		for r := 0; r < trainingInputs.Rows; r++ {
			trainErrMultiplied.Data[r] = append(trainErrMultiplied.Data[r], trainErr.Data[r][0]*tn.sigmoidDerivative(out.Data[r][0]))
		}
		trainErrMultiplied.Columns = 1

		trainingInputsTranspose := trainingInputs.Transpose()

		adjustment, err := trainingInputsTranspose.MultiplyTensor(trainErrMultiplied)
		if err != nil {
			return err
		}

		/*trainErrMultipliedTranspose := trainErrMultiplied.Transpose()
		adjustment := tensor.New(trainingInputsTranspose.Columns, 1)
		for ct := 0; ct < trainingInputsTranspose.Rows; ct++ {
			adjustment.Data[ct][0], err = tensor.DotProduct(trainingInputsTranspose.Data[ct], trainErrMultipliedTranspose.Data[0])
			if err != nil {
				return err
			}
		} */

		for c := 0; c < tn.SynapticWeights.Columns; c++ {
			tn.SynapticWeights.Data[0][c] = tn.SynapticWeights.Data[0][c] + adjustment.Data[c][0]
		}
	}

	return nil
}

// Think runs the neural network on new inputs
func (tn *Tann) Think(inputs *tensor.Tensor) (*tensor.Tensor, error) {

	err := inputs.Validate()
	if err != nil {
		return nil, err
	}
	if inputs.Columns != tn.SynapticWeights.Columns {
		return nil, fmt.Errorf("input dimensions (%d,%d) do not match synaptic weight dimensions (%d,%d)", inputs.Rows, inputs.Columns, tn.SynapticWeights.Rows, tn.SynapticWeights.Columns)
	}

	thoughts := tensor.New(inputs.Rows, 0)

	for i := 0; i < inputs.Rows; i++ {
		dotprod, err := tensor.DotProduct(inputs.Data[i], tn.SynapticWeights.Data[0])
		if err != nil {
			return nil, err
		}
		thoughts.Data[i] = append(thoughts.Data[i], tn.sigmoid(dotprod))
	}

	thoughts.Columns++

	return thoughts, nil
}
