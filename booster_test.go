package xgboost_test

import (
	"bufio"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/getumen/go-xgboost"
	"github.com/stretchr/testify/assert"
)

func TestBooster_LoadModel(t *testing.T) {
	target, err := xgboost.LoadModel("test_data/xgboost.model")
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()

	assert.Equal(t, 30, target.NumFeatures())
}

func TestBooster_LoadModelFromReader(t *testing.T) {
	f, err := os.Open("test_data/xgboost.model")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	target, err := xgboost.LoadModelFromReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()

	assert.Equal(t, 30, target.NumFeatures())
}

func TestBooster_Predict(t *testing.T) {
	nCol := 30
	var nRow int
	feature := make([]float32, 0)

	featureFile, err := os.Open("test_data/feature.csv")
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(featureFile)
	for scanner.Scan() {
		nRow++
		featureValues := strings.Split(scanner.Text(), ",")
		for _, valueString := range featureValues {
			value, err := strconv.ParseFloat(valueString, 32)
			if err != nil {
				t.Fatal(err)
			}
			feature = append(feature, float32(value))
		}
	}

	dMatrix, err := xgboost.NewDMatrixFromMat(feature, nRow, nCol, float32(math.NaN()))
	if err != nil {
		t.Fatal(err)
	}
	defer dMatrix.Close()

	expectedScores := make([]float32, 0)

	scoreFile, err := os.Open("test_data/score.csv")
	if err != nil {
		t.Fatal(err)
	}
	scanner = bufio.NewScanner(scoreFile)
	for scanner.Scan() {
		nRow++
		scoreValues := strings.Split(scanner.Text(), ",")
		for _, valueString := range scoreValues {
			value, err := strconv.ParseFloat(valueString, 32)
			if err != nil {
				t.Fatal(err)
			}
			expectedScores = append(expectedScores, float32(value))
		}
	}

	model, err := xgboost.LoadModel("test_data/xgboost.model")
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	actualScore, err := model.Predict(dMatrix, false, 0, false)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(t, len(expectedScores), len(actualScore))
	assert.InDeltaSlice(t, expectedScores, actualScore, 1e-7)
}
