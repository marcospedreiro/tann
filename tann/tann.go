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
func New(d int, n int) *Tann {

	rseed := rand.Float64()

	t := tensor.New(d, n)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
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

	if trainingInputs.Rows != tn.SynapticWeights.Rows || trainingInputs.Columns != tn.SynapticWeights.Columns {
		return fmt.Errorf("trainingInputs dimensions (%d,%d) do not match synaptic weight dimensions (%d,%d)", trainingInputs.Rows, trainingInputs.Columns, tn.SynapticWeights.Rows, tn.SynapticWeights.Columns)
	}
	if trainingOutputs.Rows != tn.SynapticWeights.Rows {
		return fmt.Errorf("trainingOutputs rows (%d) do not match synaptic weight rows (%d)", trainingOutputs.Rows, tn.SynapticWeights.Rows)
	}

	for i := 0; i < trainingIterations; i++ {

		out, err := tn.Think(trainingInputs)
		if err != nil {
			return err
		}

		tempTrainErr := tensor.New(trainingOutputs.Rows, trainingOutputs.Columns)
		tempTrainErr.Data = trainingOutputs.Data

		err = tempTrainErr.SubtractTensor(out)
		if err != nil {
			return err
		}

		for r := 0; r < trainingInputs.Rows; r++ {

			tempMulti := tensor.New(tempTrainErr.Rows, tempTrainErr.Columns)
			tempMulti.Data = tempTrainErr.Data

			tempMulti.MultiplyScalar(tn.sigmoidDerivative(out.Data[r][0]))

			tempMultiTranspose := tempMulti.Transpose()

			adjustment, err := tensor.DotProduct(trainingInputs.Data[r], tempMultiTranspose.Data[0])
			if err != nil {
				return err
			}

			for c := 0; c < tn.SynapticWeights.Columns; c++ {
				tn.SynapticWeights.Data[r][c] = tn.SynapticWeights.Data[r][c] + adjustment
			}

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
	if inputs.Rows != tn.SynapticWeights.Rows || inputs.Columns != tn.SynapticWeights.Columns {
		return nil, fmt.Errorf("input dimensions (%d,%d) do not match synaptic weight dimensions (%d,%d)", inputs.Rows, inputs.Columns, tn.SynapticWeights.Rows, tn.SynapticWeights.Columns)
	}

	thoughts := tensor.New(inputs.Rows, 1)

	for i := 0; i < inputs.Rows; i++ {
		dotprod, err := tensor.DotProduct(inputs.Data[i], tn.SynapticWeights.Data[i])
		if err != nil {
			return nil, err
		}
		thoughts.Data[i][0] = tn.sigmoid(dotprod)
	}

	return thoughts, nil
}
