package xgboost

// #cgo LDFLAGS: -lxgboost -ldmlc -lpthread -lm -lrt
// #include <stdlib.h>
// #include "xgboost/c_api.h"
import "C"
import (
	"errors"
	"log"
	"unsafe"
)

var (
	errCreateDMatrixHandle = errors.New("create dmatrix failed")
	errFreeDMatrix         = errors.New("free dmatrix failed")
)

type DMatrix struct {
	pointer C.DMatrixHandle
}

func NewDMatrixFromMat(data []float32, nrow, ncol int, missing float32) (*DMatrix, error) {
	var handle C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat(
		(*C.float)(unsafe.Pointer(&data[0])),
		C.ulong(nrow),
		C.ulong(ncol),
		C.float(missing),
		&handle,
	)
	if C.int(ret) == -1 {
		return nil, errCreateDMatrixHandle
	}
	return &DMatrix{handle}, nil
}

func (m DMatrix) Close() error {
	ret := C.XGDMatrixFree(m.pointer)
	if C.int(ret) == -1 {
		return errFreeDMatrix
	}
	return nil
}

func (m DMatrix) NumCol() int {
	var ncol uint64
	ret := C.XGDMatrixNumCol(m.pointer, (*C.ulong)(&ncol))
	if C.int(ret) == -1 {
		log.Panicf("fail to get col")
	}
	return int(ncol)
}

func (m DMatrix) NumRow() int {
	var nrow uint64
	ret := C.XGDMatrixNumRow(m.pointer, (*C.ulong)(&nrow))
	if C.int(ret) == -1 {
		log.Panicf("fail to get row")
	}
	return int(nrow)
}
