package main

import (
	"fmt"

	"github.com/marcospedreiro/tann/tann"
	"github.com/marcospedreiro/tann/tensor"
)

func main() {

	rows := 4
	columns := 3

	tnn := tann.New(columns)

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

	fmt.Printf("synaptic weights before training:\n%v\n", tnn.SynapticWeights.Data)

	err := tnn.Train(trainingData, trainingAnswers, 10000)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("synaptic weights after training:\n%v\n", tnn.SynapticWeights.Data)

	NewData := tensor.New(1, 3)
	NewData.Data = [][]float64{
		{1, 0, 0},
	}

	fmt.Printf("thinking about new data:\n%v\n", NewData.Data)

	answer, err := tnn.Think(NewData)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("answers to new data:\n%v\n", answer.Data)

	return
}
