package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello")

	a := [][]uint8{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
	}
	fmt.Println(a)
	fmt.Println(len(a[0]))

	return
}
