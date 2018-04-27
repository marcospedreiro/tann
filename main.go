package main

import (
	"fmt"

	"github.com/marcospedreiro/tann/tensor"
)

func main() {
	t := tensor.New(2, 4)
	t.Data = [][]float64{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
	}

	u := tensor.New(2, 4)
	u.Data = [][]float64{
		{8, 9, 10, 11},
		{2, 3, 4, 5},
	}

	txu, err := t.MultiplyTensor(u.Transpose())

	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(txu.Data)

	return
}
