package ann

// Ann interface for an artificial neural network
type Ann interface {
	Train([][]interface{}, [][]interface{}, int) error

	Think([][]interface{}) error
}
