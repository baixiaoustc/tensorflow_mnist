package main

import (
	"flag"
	"log"
	"strconv"

	"github.com/petar/GoMNIST"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	// "github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	_index := flag.String("index", "", "index of a mnist test data")
	flag.Parse()
	index, err := strconv.ParseInt(*_index, 10, 32)
	if err != nil {
		log.Fatal(err)
	}

	saveModel, err := tf.LoadSavedModel("/Users/baixiao/tensorflow/mnist/export/", []string{"tag"}, nil)
	if err != nil {
		log.Fatal(err)
	}

	graph := saveModel.Graph
	session := saveModel.Session
	defer session.Close()

	// tensor, err := dummyInputTensor(28 * 28)
	tensor, err := mnistTensor(int(index))
	if err != nil {
		log.Fatal(err)
	}
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("infer").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("result is", output[0].Value())
	// output[0].Value() is a vector containing probabilities of
	// labels for each image in the "batch". The batch size was 1.
	// Find the most probably label index.
	// probabilities := output[0].Value().([][]float32)[0]
	// printBestLabel(probabilities, labelsfile)
}

func dummyInputTensor(size int) (*tf.Tensor, error) {

	imageData := [][]float32{make([]float32, size)}
	for i := range imageData[0] {
		imageData[0][i] = 1
	}
	return tf.NewTensor(imageData)
}

func mnistTensor(index int) (*tf.Tensor, error) {
	_, test, err := GoMNIST.Load("/Users/baixiao/Go/src/github.com/petar/GoMNIST/data")
	if err != nil {
		log.Fatal(err)
	}

	var x []byte
	sweeper := test.Sweep()
	i := 0
	for {
		image, label, present := sweeper.Next()
		if !present {
			break
		}

		if i == index {
			log.Println("label is", label)
			x = image
			break
		}
		i++
	}

	imageData := [][]float32{make([]float32, 28*28)}
	for i := range imageData {
		for j := range imageData[i] {
			imageData[i][j] = float32(x[i*28+j])
			// log.Println(i, j, imageData[i][j])
		}
	}
	// log.Println(x)
	return tf.NewTensor(imageData)
}
