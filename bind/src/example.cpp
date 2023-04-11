/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/4/10 10:57
 * @Version: 1.0
**/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

int add(int i, int j){
    return i+j;
}

PYBIND11_MODULE(example, m){
    m.doc() = "pybind11 example";
    m.def("add", &add, "add two number");
}

/*
$ c++ -O3 -Wall -shared -std=c++11
 -fPIC $(python3 -m pybind11 --includes) example.cpp
 -o example$(python3-config --extension-suffix)
 */








