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
  
  // Read image
  if ( argc != 2 )
  {
      printf("usage: DisplayImage.out <Image_Path>\n");
      return -1;
  }
  cv::Mat img;
  img = cv::imread( argv[1], 1 );
  if ( !img.data )
  {
      printf("No image data \n");
      return -1;
  }

  // initialize Python embedding
  Py_Initialize(); 

  // add the current folder to the Python's PATH
  PyObject *sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString("."));
    
  // this macro is defined by NumPy and must be included
  import_array1(-1);
  
  // load our python script (see the gist at the bottom)
  m_PyModule = PyImport_ImportModule("pythonScript"); 
  
  if (m_PyModule != NULL)
  {
    //cout << "Module found ok" << endl;
    // get dictionary of available items in the module
    m_PyDict = PyModule_GetDict(m_PyModule);

    // grab the functions we are interested in
    m_PyFooBar = PyDict_GetItemString(m_PyDict, "foo_bar");
    
    // execute the function    
    if (m_PyFooBar != NULL)
    {
      //cout << "Function found ok" << endl;

      // Convertir image en format lisible par PyObject
      // total number of elements (here it's a grayscale 640x480)
      int nElem = img.size[0] * img.size[1];
      // create an array of apropriate datatype
      uchar* m = new uchar[nElem];
      // copy the data from the cv::Mat object into the array
      std::memcpy(m, img.data, nElem * sizeof(uchar));
      // the dimensions of the matrix
      npy_intp mdim[] = { img.size[0], img.size[1] };
      
      // convert the cv::Mat to numpy.array
      PyObject* mat = PyArray_SimpleNewFromData(2, mdim, NPY_UINT8, (void*) m);
      
      // create a Python-tuple of arguments for the function call
      // "()" means "tuple". "O" means "object"
      PyObject* args = Py_BuildValue("(O)", mat);
      
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
          //cout << "Dimensions ok" << endl;
          double **result;
          if (PyArray_AsCArray((PyObject **) &arr,
                      (void **) &result, dims, num_dims, descr) < 0){
              PyErr_SetString(PyExc_TypeError, "error converting to c array");
              return NULL;}
          printf("Taille de l'image: \n");
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