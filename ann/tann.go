package ann

import (
	"fmt"
	"math"
)

// Tann is a thing
type Tann struct {
	/*Seed the random number generator, so it generates the same numbers
	  every time the program runs.*/
	RandomSeed float64

	/*We model a single neuron, with 3 input connections and 1 output connection.
	  We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
	  and mean 0.*/
	SynapticWeights []float64
}

/* The Sigmoid function, which describes an S shaped curve.
We pass the weighted sum of the inputs through this function to normalise them between 0 and 1. */
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

/* The derivative of the Sigmoid function. This is the gradient of the Sigmoid curve.
It indicates how confident we are about the existing weight.*/
func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// Train the neural network on the training set for a determined number of iterations
func (t *Tann) Train(trainingInputs [][]interface{}, trainingOutputs []interface{}, iterations int) error {
	for i := 0; i < iterations; i++ {
		output, err := t.Think(trainingInputs)
		if err != nil {
			fmt.Println("error thinking while training:", err.Error())
			continue
		}
		fmt.Println("lol: ", output)
		//trainingError := trainingOutputs - output
	}
	return nil
}

// Think runs the neural network on new inputs
func (t *Tann) Think(inputs [][]interface{}) ([]interface{}, error) {
	return make([]interface{}, 0), nil
}
