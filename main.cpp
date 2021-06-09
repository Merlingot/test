/*
This requires adding the "include" directory of your python to the include diretories 
of your project, e.g., in Visual Studio youd add `C:\Program Files\Python36\include`. 
You also need to add the 'include' directory of your NumPy package, e.g. 
`C:\Program Files\PythonXX\Lib\site-packages\numpy\core\include`.
Additionally, you need to link your "python3#.lib" library, e.g. `C:\Program Files\Python3X\libs\python3X.lib`. 
*/

// python bindings
#include "Python.h"
#include "numpy/arrayobject.h"
#include <iostream>
#include <ostream>
#include "opencv2/opencv.hpp"

using namespace std;

// for the references to all the functions
PyObject *m_PyDict, *m_PyFooBar;

// for the reference to the Pyhton module
PyObject* m_PyModule;

int main(int argc, char** argv) {
  // 
  // initialize Python embedding
  
  Py_Initialize(); 
  
  // set the command line arguments (can be crucial for some python-packages, like tensorflow)
  // PySys_SetArgv(argc, (wchar_t**)argv);
  
  // add the current folder to the Python's PATH
  PyObject *sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString("."));
    
  // this macro is defined by NumPy and must be included
  import_array1(-1);
  
  // load our python script (see the gist at the bottom)
  m_PyModule = PyImport_ImportModule("pythonScript"); 
  
  if (m_PyModule != NULL)
  {
    cout << "File pythonScript loaded ok" << endl;
    // get dictionary of available items in the module
    m_PyDict = PyModule_GetDict(m_PyModule);

    // grab the functions we are interested in
    m_PyFooBar = PyDict_GetItemString(m_PyDict, "foo_bar");
    
    // execute the function    
    if (m_PyFooBar != NULL)
    {
      cout << "Function foo_bar found ok" << endl;

      // CREATE A cv:MAT FOR TEST -------------------------------------------
      // take a cv::Mat object from somewhere (we'll just create one)
      cv::Mat img = cv::Mat::zeros(480, 640, CV_8U);
      
      // total number of elements (here it's a grayscale 640x480)
      int nElem = 640 * 480;
      
      // create an array of apropriate datatype
      uchar* m = new uchar[nElem];
      
      // copy the data from the cv::Mat object into the array
      std::memcpy(m, img.data, nElem * sizeof(uchar));
      
      // the dimensions of the matrix
      npy_intp mdim[] = { 480, 640 };
      
      // convert the cv::Mat to numpy.array
      PyObject* mat = PyArray_SimpleNewFromData(2, mdim, NPY_UINT8, (void*) m);
      
      // create a Python-tuple of arguments for the function call
      // "()" means "tuple". "O" means "object"
      PyObject* args = Py_BuildValue("(O)", mat);
      // if we want several arguments, we can write ("i" means "integer"):
      // PyObject* args = Py_BuildValue("(OOi)", mat, mat, 123);
      // see detailed explanation here: https://docs.python.org/2.0/ext/buildValue.html 
      
      // execute the function
      PyObject* obj = NULL;
      obj = PyObject_CallObject(m_PyFooBar, args);

      // Verification 
      if (obj==NULL) 
        return NULL;
    
      
      // Transformation en PyArray
      int typenum = NPY_DOUBLE;
      PyObject *arr =  PyArray_FROM_OTF( obj, typenum, NPY_ARRAY_INOUT_ARRAY);

      // Transformer en C array
      // get number of dimensions:
      npy_intp num_dims = PyArray_NDIM(arr);
      npy_intp *num_len = PyArray_DIMS(arr);
      // help-vars
      PyArray_Descr *descr = PyArray_DescrFromType(typenum);
      npy_intp dims[num_dims];

      // incref, as PyArray_AsCArray steals reference
      Py_INCREF(arr);

      // is 2D-array (gives segmentation fault otherwise)
      if (num_dims == 2 ){
          double **result;
          if (PyArray_AsCArray((PyObject **) &arr,
                      (void **) &result, dims, num_dims, descr) < 0){
              PyErr_SetString(PyExc_TypeError, "error converting to c array");
              return NULL;}
          printf("Elements of array: \n");
          for (int i=0; i < *num_len; i++)
            printf("%.2f\n", *result[0,i]);
          
          // free C-like array
          PyArray_Free((PyObject *) arr, (void *) result);
      }
        
      // decrement the object references
      Py_XDECREF(mat);
      Py_XDECREF(args);
    
      delete[] m;
    }
  }
  else
  {
    std::cerr << "Failed to load the Python module!" << std::endl;
    PyErr_Print();
  }
  
  return 0;
}