package main

import (
	"fmt"

	"github.com/marcospedreiro/tann/tann"
	"github.com/marcospedreiro/tann/tensor"
)

func main() {

	rows := 4
	columns := 3

	tnn := tann.New(rows, columns)

	trainingData := tensor.New(rows, columns)
	trainingData.Data = [][]float64{
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 1},
		{0, 1, 1},
	}

	trainingAnswers := tensor.New(rows, 1)
	trainingAnswers.Data = [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	fmt.Printf("Synaptic weights before training: %v\n", tnn.SynapticWeights.Data)

	err := tnn.Train(trainingData, trainingAnswers, 10000)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("Synaptic weights after training: %v\n", tnn.SynapticWeights.Data)

	NewData := tensor.New(rows, columns)
	NewData.Data = [][]float64{
		{1, 0, 0},
		{1, 1, 1},
		{0, 0, 0},
		{0, 1, 1},
	}

	answer, err := tnn.Think(NewData)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("\n")
	fmt.Printf("answers to new data: %v\n", answer.Data)

	return
}
