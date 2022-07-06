package xgboost

// #cgo LDFLAGS: -lxgboost -ldmlc -lpthread -lm -lrt
// #include <stdlib.h>
// #include "xgboost/c_api.h"
import "C"
import (
	"errors"
	"fmt"
	"io"
	"log"
	"unsafe"
)

var (
	errCreateBooster = errors.New("create booster failed")
	errLoadModel     = errors.New("load model failed")
	errPredict       = errors.New("predict failed")
	errFreeBooster   = errors.New("free booster failed")
)

type Booster struct {
	pointer C.BoosterHandle
}

func LoadModel(fileName string) (*Booster, error) {
	var handle C.BoosterHandle
	ret := C.XGBoosterCreate(
		nil,
		0,
		&handle,
	)
	if C.int(ret) == -1 {
		return nil, errCreateBooster
	}
	ret = C.XGBoosterLoadModel(
		handle,
		C.CString(fileName),
	)
	if C.int(ret) == -1 {
		return nil, errLoadModel
	}
	return &Booster{handle}, nil
}

func LoadModelFromReader(reader io.Reader) (*Booster, error) {
	var handle C.BoosterHandle
	ret := C.XGBoosterCreate(
		nil,
		0,
		&handle,
	)
	if C.int(ret) == -1 {
		return nil, errCreateBooster
	}

	blob, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("fail to read reader: %w", err)
	}

	ret = C.XGBoosterLoadModelFromBuffer(
		handle,
		C.CBytes(blob),
		C.ulong(len(blob)),
	)
	if C.int(ret) == -1 {
		return nil, errLoadModel
	}
	return &Booster{handle}, nil
}

func (b *Booster) Close() error {
	ret := C.XGBoosterFree(b.pointer)
	if C.int(ret) == -1 {
		return errFreeBooster
	}
	return nil
}

func (b *Booster) Predict(
	dmat *DMatrix,
	outputMargin bool,
	ntreeLimit int,
	training bool,
) ([]float32, error) {
	var optionMask int32 = 0
	if outputMargin {
		optionMask = 1
	}
	var training_ int
	if training {
		training_ = 2
	} else {
		training_ = 1
	}

	var (
		outLen    C.ulong
		outResult *C.float
	)
	ret := C.XGBoosterPredict(
		b.pointer,
		dmat.pointer,
		C.int(optionMask),
		C.unsigned(ntreeLimit),
		C.int(training_),
		&outLen,
		&outResult,
	)
	if C.int(ret) == -1 {
		return nil, errPredict
	}
	length := int(outLen)
	result := make([]float32, length)
	for i, v := range unsafe.Slice(outResult, length) {
		result[i] = float32(v)
	}
	return result, nil
}

func (b *Booster) NumFeatures() int {
	var numFeatures uint64
	ret := C.XGBoosterGetNumFeature(b.pointer, (*C.ulong)(&numFeatures))
	if C.int(ret) == -1 {
		log.Panicf("fail to get num_features")
	}
	return int(numFeatures)
}
