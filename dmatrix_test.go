package xgboost_test

import (
	"testing"

	"github.com/getumen/go-xgboost"
	"github.com/stretchr/testify/assert"
)

func TestLoadModel(t *testing.T) {

	data := []float32{0, 1, 2, 3, 4, 5}
	nrow := 2
	ncol := 3
	target, err := xgboost.NewDMatrixFromMat(data, nrow, ncol, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()

	assert.Equal(t, nrow, target.NumRow())
	assert.Equal(t, ncol, target.NumCol())
}
