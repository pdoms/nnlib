#![allow(unused)]
use core::fmt;
use std::alloc;
use std::f64::consts::E;
use std::ops::{Index, Range, Sub, SubAssign, AddAssign, Add, Mul, Div, Neg};
use rand::distributions::uniform::{SampleRange, SampleUniform};
use rand::{self, Rng};



pub struct Matrix<T> {
    pub cols: usize,
    pub rows: usize,
    pub stride: usize,
    pub ptr: *mut T,
    pub len: usize,
}
///(rows, cols)
type MatIdx = (usize, usize);


impl<T: fmt::Debug + fmt::Display + Add + Sub + AddAssign + Mul + Div + Neg + Copy + PartialEq + PartialOrd + Default + Mul<Output = T>> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::from("[ ");
        if self.is_vector() {
            for i in 0..self.len {
                s = format!("{s}{}, ", self.get((0, i)));
            }
            s.pop();
            s.pop();
            s = format!("{s}]\n");
            f.write_str(s.as_str())
        } else {
            let mut s = String::from("[\n");
            for i in 0..self.rows {
                s = format!("{s}\t[");
                for j in 0..self.cols {
                    s = format!("{s}{}, ", self.get((i, j)));
                }
                s.pop();
                s.pop();
                s = format!("{s}],\n");
            }
            s.pop();
            s.pop();
            s = format!("{s}\n]\n");
            f.write_str(s.as_str())
        }
    }
}

impl<T: fmt::Debug + fmt::Display + Add + AddAssign + Sub + Mul + Div + Neg + Copy + PartialEq + PartialOrd + Default + Mul<Output = T>> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::from("[");
        if self.is_vector() {
            for i in 0..self.len {
                s = format!("{s}{}, ", self.get((0, i)));
            }
            s.pop();
            s.pop();
            s = format!("{s}]\n");
            f.write_str(s.as_str())
        } else {
            let mut s = String::from("[\n");
            for i in 0..self.rows {
                s = format!("{s}\t[");
                for j in 0..self.cols {
                    s = format!("{s}{}, ", self.get((i, j)));
                }
                s.pop();
                s.pop();
                s = format!("{s}],\n");
            }
            s.pop();
            s.pop();
            s = format!("{s}\n]\n");
            f.write_str(s.as_str())
        }
    }
}


impl<T: PartialEq> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.cols == other.cols && self.rows == other.rows && self.len == other.len {
        unsafe {
            let self_slice = std::slice::from_raw_parts(self.ptr, self.len);
            let other_slice = std::slice::from_raw_parts(other.ptr, other.len);
            self_slice == other_slice
        }
        } else {
            false
        }
    }
}

impl<T: PartialEq> Eq for Matrix<T> {}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        unsafe {
            let mem_size =std::mem::size_of::<T>(); 
            let layout = alloc::Layout::from_size_align_unchecked(                  
                mem_size*self.len,
                std::mem::align_of::<T>());
            alloc::dealloc(self.ptr as *mut u8, layout)
        }
    }
}

impl<T> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        let mem_size =std::mem::size_of::<T>(); 
        let layout = alloc::Layout::array::<T>(self.len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc(layout)
        } as *mut T;
        unsafe {
            std::ptr::copy(self.ptr, ptr, self.len*mem_size)
        };
        Self {
            cols: self.cols,
            rows: self.rows,
            stride: self.stride,
            len: self.len,
            ptr,
        }
    }
}

impl<T: Add + AddAssign + Mul + Div + Neg + Copy + PartialEq + PartialOrd + Default + Mul<Output = T>> Matrix<T> {
    /// rows, cols
    pub fn new(index: MatIdx) -> Self {
        let mem_size =std::mem::size_of::<T>(); 
        let (rows, cols) = index;
        let mut len = cols*rows;
        if rows == 0 {
            len = cols*1;
        }
        let layout = alloc::Layout::array::<T>(len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc(layout)
        } as *mut T;
        Self {
            cols,
            rows,
            stride: cols,
            ptr,
            len
        }
    }

    /// Creates a [`Matrix`] with the shape of the provided Matrix.
    pub fn like(mat: &Matrix<T>) -> Self {
        Self::new(mat.shape())
    }

    /// Creates a [`Matrix`] with the shape of the provided Matrix and fills
    /// it with the provided value.
    /// Use this for type casting!
    pub fn like_with(mat: &Matrix<T>, v: T) -> Self {
        let m = Self::like(mat);
        unsafe {
            for i in 0..m.len {
                m.ptr.add(i).write(v);
            }
        }
        m
    }
 
    pub fn zeroed(index: MatIdx) -> Self {
        let mem_size =std::mem::size_of::<T>(); 
        let (rows, cols) = index;
        let mut len = cols*rows;
        if rows == 0 {
            len = cols*1;
        }
        let layout = alloc::Layout::array::<T>(len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc_zeroed(layout)
        } as *mut T;
        Self {
            cols,
            rows,
            stride: cols,
            ptr,
            len
        }
    }

    

    pub fn indices_one_hot(indices: &Matrix<T>, n: usize) -> Matrix<T> {
        todo!()
    }

    pub fn from_vec(mut data: Vec<T>) -> Self {
        let len = data.len();
        let mem_size =std::mem::size_of::<T>(); 
        let layout = alloc::Layout::array::<T>(len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc(layout)
        } as *mut T;
        unsafe {ptr.copy_from(data.as_mut_ptr(), len)};
        Self {
            cols: len,
            rows: 0,
            stride: 1,
            ptr,
            len
        }
    }

    pub fn from_vec2(data: Vec<Vec<T>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        let matrix = Self::zeroed((rows, cols));
        let mut flattened: Vec<T> = data.into_iter().flatten().collect();
        unsafe {
            matrix.ptr.copy_from(flattened.as_mut_ptr(), matrix.len);
        }
        matrix
    }

    pub fn shape(&self) -> MatIdx {
        (self.rows, self.cols)
    }

    pub fn is_vector(&self) -> bool {
        self.rows == 0
    }

    pub fn get(&self, index: MatIdx) -> T {
        let (rows, cols) = index;
        if self.rows == 0 {
            assert!(cols < self.cols, "index out of range");
        } else {
            assert!(rows < self.rows && cols < self.cols, "index out of range");
        }
        unsafe {
            self.ptr.add(rows*self.stride + cols).read()
        }
    }

    pub fn set(&self, f: T, index: MatIdx) {
        let (rows, cols) = index;
        if self.rows == 0 {
            assert!(cols < self.cols,  "index out of range");
        } else {
            assert!(rows < self.rows && cols < self.cols,  "index out of range");
        }
        unsafe {
            self.ptr.add(rows*self.stride + cols).write(f)
        }
    }

    fn get_row(&self, row: usize) -> Matrix<T> {
        if self.is_vector() {
            return self.clone()
        }
        assert!(row < self.rows, "index out of range");
        let mat = Matrix::new((0, self.cols));
        unsafe {
            let offset = self.ptr.add(row*self.stride);
            for i in 0..self.cols {
                mat.set(offset.add(i).read(), (0, i))
            }
        };
        mat
    }

    fn set_row(&self, i: usize, row: &Matrix<T>) {
        assert!(row.is_vector(), "src is not a vector");
        assert!(self.cols == row.cols, "cols don't match");
        self.bound_check_rows(i);
        unsafe {
            let offset = self.ptr.add(i*self.stride);
            for c in 0..row.cols {
                offset.add(c).write(row.get((0, c)));
            } 
        }
    }

    fn get_col(&self, col: usize, keepdims: bool) -> Matrix<T> {
        assert!(col < self.cols, "index out of range");
        assert!(!self.is_vector(), "column can only be taken from 2D array, this is a vector");
        if !keepdims {
            let mat = Matrix::new((0, self.rows));
            let mut idx = col;
            for i in 0..self.rows {
                let val = unsafe {
                    self.ptr.add(idx).read()
                };
                mat.set(val, (0, i));
                idx += self.stride;
            }
            return mat
        } else {
            let mat = Matrix::new((self.rows, 1));
            let mut idx = col;
            for i in 0..self.rows {
                let val = unsafe {
                    self.ptr.add(idx).read()
                };
                mat.set(val, (i, 0));
                idx += self.stride;
            }
            return mat
        }
    }

    fn set_col(&self, i: usize, col: &Matrix<T>) {
        if col.is_vector() {
            assert!(self.rows == col.cols, "rows don't match");
            self.bound_check_cols(i);
            for j in 0..self.rows {
                let v = col.get((0, j));
                self.set(v, (j, i));
            }
        } else {
            assert!(self.cols == col.cols, "rows don't match");
            for j in 0..self.rows {
                let v = col.get((j, 0));
                self.set(v, (j, i));
            }
        }
    }


    pub fn transpose(&mut self) {
        //new matrix
        let m = Matrix::<T>::new((self.cols, self.rows));
        for j in 0..self.cols {
            //get column of self 
            let col = self.get_col(j, false);
            //set it as row in new matrix
            m.set_row(j, &col);
        }
        //switch self.cols with self.rows
        let col_copy = self.cols;
        self.cols = self.rows;
        self.rows = col_copy;
        //copy new matrix data to self.ptr data
        let mem_size = std::mem::size_of::<T>(); 
        unsafe {
            std::ptr::copy(m.ptr, self.ptr, self.len*mem_size);
        };

    }

    pub fn flatten(&mut self) {
        self.cols = self.len;
        self.rows = 0;
        self.stride = 1;
    }
    pub fn copy_flat(&self) -> Matrix<T> {
        let m = Matrix::<T>::new((0, self.len));
        let mem_size = std::mem::size_of::<T>(); 
        unsafe {
            std::ptr::copy(self.ptr, m.ptr, self.len*mem_size);
        };
        m
    }

    pub fn reshape(&mut self, shape: (isize, isize)) {
        let (mut rows, mut cols) = shape;
        if (rows == -1 && cols > 0) {
            rows = self.len as isize / cols;
            assert!(rows*cols == self.len as isize, "cannot reshape Matrix of '{:?}' into shape ({rows}, {cols})", self.shape());
            self.cols = cols as usize;
            self.rows = rows as usize;
            self.stride = cols as usize;
        } else if cols == -1 && rows >= 0 {
            cols = self.len as isize / rows; 
            assert!(rows*cols == self.len as isize, "cannot reshape Matrix of '{:?}' into shape ({rows}, {cols})", self.shape());
            self.cols = cols as usize;
            self.rows = rows as usize;
            self.stride = cols as usize;
        } else if rows >= 0 && cols > 0 {
            assert!(rows*cols == self.len as isize, "cannot reshape Matrix of '{:?}' into shape ({rows}, {cols})", self.shape());
            self.cols = cols as usize;
            self.rows = rows as usize;
            self.stride = cols as usize;
        } else {
            panic!("cannot reshape Matrix of '{:?}' into shape ({rows}, {cols})", self.shape());
        } 
    }


    pub fn each<F>(&self, f: F) 
    where
        F: Fn(T) -> T
    {
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0,j));
                let adjusted = f(v);
                self.set(adjusted, (0,j))
            }
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i,j));
                    let adjusted = f(v);
                    self.set(adjusted, (i,j))
                }
            }
        }
    }


    pub fn map<F>(&self, f: F) -> Matrix<T>
    where
        F: Fn(T) -> T
    {
        let result = Matrix::like(self);
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0,j));
                let adjusted = f(v);
                result.set(adjusted, (0,j))
            }
            result
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i,j));
                    let adjusted = f(v);
                    result.set(adjusted, (i,j))
                }
            }
            result
        }
    }

    pub fn every<F>(&self, f: F) -> bool 
    where 
        F: Fn(T) -> bool
    {
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0, j));
                if !f(v) {
                    return false
                }
            }
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i, j));
                    if !f(v) {
                        return false
                    }
                }
            }
        }
        true
    }

    pub fn clip(&self, low: T, high: T) {
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0,j));
                if v <= low {
                    self.set(low, (0,j));    
                } 
                if v >= high {
                    self.set(high, (0,j));    
                }  
            }
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i,j));
                    if v <= low {
                        self.set(low, (i,j));    
                    } 
                    if v >= high {
                        self.set(high, (i,j));    
                    }  
                }
            }
        }
    }

    pub fn max(&self) -> T {
        let mut max = T::default();
        unsafe {
            for i in 0..self.len {
                let v = self.ptr.add(i).read();    
                if i == 0 {
                    max = v
                } else if v > max {
                    max = v
                }
            }
        };
        max
    }

    pub fn max_axis(&self, axis: usize, keepdims: bool) -> Matrix<T> {
        match axis {
           0 => {
                let mut result = Matrix::new((1, self.cols));
                for i in 0..self.cols {
                    let col = self.get_col(i, false);
                    result.set(col.max(), (0, i));
                }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            1 => {
                let mut result = Matrix::new((self.rows, 1));
                for i in 0..self.rows {
                    let col = self.get_row(i);
                    result.set(col.max(), (i, 0));
                }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            _ => {
                panic!("axis {axis} does not exist on Matrix")
            }
        }
    }

    pub fn argmax(&self) -> MatIdx {
        let mut max = T::default();
        let mut idx: MatIdx = (0, 0);
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0, j));
                if j == 0 {
                    max = v;
                    idx = (0, j);
                } else if v > max {
                    max = v;
                    idx = (0, j);
                }
            }
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i, j));
                    if i == 0 && j == 0 {
                        max = v;
                        idx = (i, j);
                    } else if v > max {
                        max = v;
                        idx = (i, j);
                    }
                }
            }
        }
        idx
    }

    pub fn argmax_axis(&self, axis: usize) -> Matrix<i32> {
        match axis {
           0 => {
                let mut result: Vec<i32> = vec![];
                for i in 0..self.cols {
                    let col = self.get_col(i, false);
                    result.push(col.argmax().1 as i32);
                }
                Matrix::from_vec(result)
            },
            1 => {
                let mut result: Vec<i32> = vec![];
                for i in 0..self.rows {
                    let row = self.get_row(i);
                    result.push(row.argmax().1 as i32)
                }
                Matrix::from_vec(result)
            },
            _ => {
                panic!("axis {axis} does not exist on Matrix")
            }
        }
    } 

    pub fn sum(&self) -> T {
        let mut acc = T::default();
        unsafe {
            for i in 0..self.len {
                acc += self.ptr.add(i).read();    
            }
        };
        acc
    }

    pub fn sum_axis(&self, axis: usize, keepdims: bool) -> Matrix<T> {
        match axis {
           0 => {
                let mut result = Matrix::new((1, self.cols));
                for i in 0..self.cols {
                    let col = self.get_col(i, false);
                    result.set(col.sum(), (0, i));
                }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            1 => {
                let mut result = Matrix::new((self.rows, 1));
                for i in 0..self.rows {
                    let col = self.get_row(i);
                    result.set(col.sum(), (i,0));
                }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            _ => {
                panic!("axis {axis} does not exist on Matrix")
            }
        }
    }

    pub fn v_dot(&self, other: &Matrix<T>) -> T {
        assert!(self.is_vector(), "'self' needs to be a vector (0, 1)");
        assert!(other.is_vector(), "'other' needs to be a vector (0, 1)");
        assert!(self.len == other.len, "both vectors need to have the same length");
        let mut acc = T::default();
        for i in 0..self.cols {
            let v1: T = self.get((0, i));
            let v2: T = other.get((0, i));
            acc += v1 * v2;
        }
        acc
    }
    
    
    pub fn mat_mul(&self, other: &Matrix<T>) -> Matrix<T> {
        if let Some(shape) = self.prepare_mat_mul(other) {
            let trg = Matrix::<T>::new(shape);
            for i in 0..self.rows {
                let v1 = self.get_row(i);
                for j in 0..other.cols {
                    let v2 = other.get_col(j, false);
                    let dot = v1.v_dot(&v2);
                    trg.set(dot, (i,j));
                }
            }
            trg
        } else {
            panic!("M1 cols ({}) don't match rows ({})", self.cols, other.rows)
        }
    }
    
    pub fn dot(&self, other: &Matrix<T>) -> Matrix<T> {
        if !self.is_vector() && !other.is_vector() {
            self.mat_mul(other)
        } else {
            assert!(other.is_vector(), "'other' needs to be a vector");
            assert!(self.cols == other.cols, "'self' cols don't match 'other' rows");
            let result = Matrix::new((0, self.rows));
            for i in 0..self.rows {
                let row = self.get_row(i);
                result.set(row.v_dot(other), (0, i))
            }
            result
        }
    }
    
    fn prepare_mat_mul(&self, other: &Matrix<T>) -> Option<MatIdx> {
        if self.cols == other.rows {
            Some((self.rows, other.cols))
        } else {
            None
        }
    }
    
    fn shape_match(&self, other: &Matrix<T>) -> bool {
        self.cols == other.cols && self.rows == other.rows
    }
    fn col_match(&self, other: &Matrix<T>) -> bool {
        self.cols == other.cols
    }
    fn row_match(&self, other: &Matrix<T>) -> bool {
        self.rows == other.rows
    }
    
    fn bound_check_cols(&self, i: usize) {
        if i >= self.cols {
            let cols = self.cols;
            panic!("col index out of range [{i} >= {cols}]");
        }
    }
    fn bound_check_rows(&self, i: usize) {
        if i >= self.rows {
            let rows = self.rows;
            panic!("row index out of range [{i} >= {rows}]");
        }
    }
}

impl Matrix<f64> {
    pub fn ln(&self) {
        self.each(|x| x.ln())
    }

    pub fn ln_copy(&self) -> Matrix<f64> {
        let m = self.clone();
        m.ln();
        m
    }

    pub fn neg_ln(&self) {
        self.each(|x| -x.ln())
    }

    pub fn neg_ln_copy(&self) -> Matrix<f64> {
        let m = self.clone();
        m.neg_ln();
        m
    }

    pub fn exp(&self) {
        self.each(|x| E.powf(x))
    }

    pub fn exp_copy(&self) -> Matrix<f64> {
        let m = self.clone();
        m.exp();
        m
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.len as f64
    }
    
    pub fn mean_axis(&self, axis: usize, keepdims: bool) -> Matrix<f64> {
        match axis {
           0 => {
                let mut result = Matrix::new((1, self.cols));
                    for i in 0..self.cols {
                        let col = self.get_col(i, false);
                        result.set(col.mean(), (0, i));
                    }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            1 => {
                let mut result = Matrix::new((self.rows, 1));
                for i in 0..self.rows {
                    let col = self.get_row(i);
                    result.set(col.mean(), (i, 0));
                }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            _ => {
                panic!("axis {axis} does not exist on Matrix")
            }
        }
    }
}

impl Matrix<i32> {
    pub fn mean(&self) -> f64 {
        f64::from(self.sum()) / self.len as f64
    }
    
    pub fn mean_axis(&self, axis: usize, keepdims: bool) -> Matrix<f64> {
        match axis {
           0 => {
                let mut result = Matrix::new((1, self.cols));
                    for i in 0..self.cols {
                        let col = self.get_col(i, false);
                        result.set(col.mean(), (0, i));
                    }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            1 => {
                let mut result = Matrix::new((self.rows, 1));
                for i in 0..self.rows {
                    let col = self.get_row(i);
                    result.set(col.mean(), (i, 0));
                }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            _ => {
                panic!("axis {axis} does not exist on Matrix")
            }
        }
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = MatrixIntoIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixIntoIterator {
            matrix: self,
            index: 0
        }
    }

}

pub struct MatrixIntoIterator<T> {
    matrix: Matrix<T>,
    index: usize
}

impl<T> Iterator for MatrixIntoIterator<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.matrix.len {
            let v = unsafe {
                self.matrix.ptr.add(self.index).read()
            };
            self.index += 1;
            Some(v)
        } else {
            None
        }
    }
}


impl std::convert::From<Matrix<i32>> for Matrix<f64> {
    fn from(m: Matrix<i32>) -> Self {
        let conv = Matrix::<f64>::new(m.shape());
        for i in 0..m.rows {
            for j in 0..m.cols {
                conv.set(f64::from(m.get((i,j))), (i,j));
            }
        }
        conv
    }
}

impl<T: Add + AddAssign + Sub  + Mul + Div + Neg + Copy + SampleUniform + PartialEq + PartialOrd + Default + Mul<Output = T>>  Matrix<T> {
pub fn random(shape: MatIdx,  range: Range<T>) -> Self {
    let matrix = Self::zeroed(shape);
    let mut rng = rand::thread_rng();
    unsafe {
        for i in 0..matrix.len {
            let r: T = rng.gen_range(range.clone());
            matrix.ptr.add(i).write(r);
        }
    }
    matrix
}
}




fn max<T: PartialOrd + PartialOrd>(a: T, b: T) -> T {
if a >= b {
    a
} else {
    b
}
}



impl<T: Sub<Output = T> + fmt::Display + fmt::Debug + Default + Copy + Add + AddAssign + Mul + Div + Neg + PartialEq + PartialOrd + Default + Mul<Output = T>> Sub for Matrix<T> {

type Output = Matrix<T>;

fn sub(self, rhs: Self) -> Matrix<T> {
    if self.shape_match(&rhs) {
        let result = Self::new((self.rows, self.cols));
        for i in 0..self.rows {
            for j in 0..self.cols {
                let left = self.get((i,j));
                let right = rhs.get((i, j));
                result.set(left - right, (i, j))
            }
        }
        result
    } else if self.col_match(&rhs) && rhs.is_vector() {
        let result = Self::new((self.rows, self.cols));
        for i in 0..self.rows {
            for j in 0..self.cols {
                let left = self.get((i,j));
                let right = rhs.get((0, j));
                result.set(left - right, (i, j))
            }
        }
        result
    } else if self.row_match(&rhs) && rhs.cols == 1 {
        let result = Self::new((self.rows, self.cols));
        for i in 0..self.cols {
            let col = self.get_col(i, false);
            for j in 0..self.rows {
                let left = col.get((0,j));
                let right = rhs.get((j, 0));
                col.set(left - right, (0, j));
            }
            result.set_col(i, &col)
        }
        result
    } else {
        panic!("cannot match minuend and subtrahend")
    }
}

}



impl<T: Add<Output = T> + fmt::Display + fmt::Debug + Default + Copy + AddAssign + Sub + Mul + Div + Neg + PartialEq + PartialOrd + Default + Mul<Output = T>> Add for Matrix<T> {
type Output = Self;
fn add(self, rhs: Self) -> Self::Output {
    if self.shape_match(&rhs) {
        let result = Matrix::new((self.rows, self.cols));
        for i in 0..self.rows {
            for j in 0..self.cols {
                let left = self.get((i,j));
                let right = rhs.get((i, j));
                result.set(left + right, (i, j))
            }
        }
        result
    } else if self.col_match(&rhs) && rhs.is_vector() {
        let result = Matrix::new((self.rows, self.cols));
        for i in 0..self.rows {
            for j in 0..self.cols {
                let left = self.get((i,j));
                let right = rhs.get((0, j));
                result.set(left + right, (i, j))
            }
        }
        result
    } else if self.row_match(&rhs) && rhs.cols == 1 {
        let result = Matrix::new((self.rows, self.cols));
        for i in 0..self.cols {
            let col = self.get_col(i, false);
            for j in 0..self.rows {
                let left = col.get((0,j));
                let right = rhs.get((j, 0));
                col.set(left + right, (0, j));
            }
            result.set_col(i, &col)
        }
        result
    } else {
        panic!("cannot match addends")
    }
}
} 

impl<T> Mul<Matrix<T>> for Matrix<T> 
    where
        T: Mul<Output = T> + fmt::Display + fmt::Debug + Default + Copy + Add + AddAssign + Sub + Div + Neg + PartialEq + PartialOrd + Default 
{
    type Output = Matrix<T>;
    fn mul(self, rhs: Self) -> Matrix<T> {
        if self.shape_match(&rhs) {
            let result = Matrix::new((self.rows, self.cols));
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let left = self.get((i,j));
                    let right = rhs.get((i, j));
                    result.set(left * right, (i, j))
                }
            }
            result
        } else if self.col_match(&rhs) && rhs.is_vector() {
            let result = Matrix::new((self.rows, self.cols));
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let left = self.get((i,j));
                    let right = rhs.get((0, j));
                    result.set(left * right, (i, j))
                }
            }
            result
        } else if self.row_match(&rhs) && rhs.cols == 1 {
            let result = Matrix::new((self.rows, self.cols));
            for i in 0..self.cols {
                let col = self.get_col(i, false);
                for j in 0..self.rows {
                    let left = col.get((0,j));
                    let right = rhs.get((j, 0));
                    col.set(left * right, (0, j));
                }
                result.set_col(i, &col)
            }
            result
        } else {
            panic!("cannot match addends")
        }
    }
}

impl<T: Div<Output = T> + fmt::Display + fmt::Debug + Default + Copy + Add + AddAssign + Sub + Mul + Neg + PartialEq + PartialOrd + Default + Mul<Output = T>> Div for Matrix<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        if self.shape_match(&rhs) {
            let result = Matrix::new((self.rows, self.cols));
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let left = self.get((i,j));
                    let right = rhs.get((i, j));
                    result.set(left / right, (i, j))
                }
            }
            result
        } else if self.col_match(&rhs) && rhs.is_vector() {
            let result = Matrix::new((self.rows, self.cols));
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let left = self.get((i,j));
                    let right = rhs.get((0, j));
                    result.set(left / right, (i, j))
                }
            }
            result
        } else if self.row_match(&rhs) && rhs.cols == 1 {
            let result = Matrix::new((self.rows, self.cols));
            for i in 0..self.cols {
                let col = self.get_col(i, false);
                for j in 0..self.rows {
                    let left = col.get((0,j));
                    let right = rhs.get((j, 0));
                    col.set(left / right, (0, j));
                }
                result.set_col(i, &col)
            }
            result
        } else {
            panic!("cannot match addends")
        }
    }
} 


impl<T: Neg<Output = T> + fmt::Display + fmt::Debug + Default + Copy + Add + AddAssign + Sub + Mul + Div + PartialEq + PartialOrd + Default + Mul<Output =T>> Neg for Matrix<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0,j));
                self.set(-v, (0, j));
            }
            self
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i,j));
                    self.set(-v, (i, j));
                }
            }
            self
        }
    }
} 
 
 
 
#[cfg(test)]
mod test {
    use super::*;
  
    #[test]
    fn la_matrix_creation() {
        Matrix::<f32>::new((2,2));
        Matrix::<i32>::new((2,2));
        Matrix::<f32>::zeroed((2,2));
        let mat = Matrix::<i32>::zeroed((2, 3));
        assert_eq!(mat.shape(), (2,3));
        let mat = Matrix::<i32>::from_vec(vec![1,2,3]);
        assert_eq!(mat.shape(), (0, 3));
        assert!(mat.is_vector());
        let mat = Matrix::<i32>::from_vec2(vec![vec![1,1], vec![1,1]]);
        assert_eq!(mat.shape(), (2,2));
        let liked = Matrix::like_with(&mat, 2);
        let liked_r = Matrix::from_vec2(vec![vec![2,2], vec![2,2]]);
        assert_eq!(liked, liked_r);
        let int_rand = Matrix::<i32>::random((4,5), -1..1);
        let float_rand = Matrix::<f32>::random((4,5), -1.0..1.0);
        let r1 = int_rand.get((1,1));
        assert!(r1 >= -1 && r1 < 1);
        let r2 = float_rand.get((2,3));
        assert!(r2 > -1.0 && r2 < 1.0);
    }

    #[test]
    fn la_matrix_get_set() {
        let mat = Matrix::<f32>::zeroed((2,1));
        mat.set(1.0, (1,0));
        assert_eq!(mat.get((1,0)), 1.0);
        assert_eq!(mat.get((0,0)), 0.0);
        let mat = Matrix::<i32>::zeroed((4,5));
        let row = Matrix::<i32>::from_vec(vec![1,2,3,4,5]);
        mat.set_row(2, &row);
        assert_eq!(mat.get_row(2), row);
        let col = Matrix::<i32>::from_vec(vec![1,2,3,4]);
        mat.set_col(2, &col);
        assert_eq!(mat.get_col(2, false), col);
        assert_eq!(mat.get((2,2)), 3);
        let with_dims = Matrix::<i32>::from_vec2(vec![vec![1],vec![2],vec![3],vec![4]]);
        assert_eq!(mat.get_col(2, true), with_dims);
    }

    #[test]
    fn la_matrix_subtraction() {
        let sub = vec![1,2,3,4];
        let mat1 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let mat2 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let res = Matrix::<i32>::zeroed((3,4));
        assert_eq!(mat1.clone() - mat2, res);
        let row_match = Matrix::<i32>::from_vec(vec![1,1,1,1]);
        let res = vec![0,1,2,3];
        let row_result = Matrix::<i32>::from_vec2(vec![res.clone(), res.clone(), res.clone()]);
        assert_eq!(mat1.clone() - row_match, row_result);
        let col_match = Matrix::<i32>::from_vec2(vec![vec![1],vec![2],vec![3]]);
        let col_result = Matrix::<i32>::from_vec2(vec![res, vec![-1, 0, 1, 2], vec![-2, -1, 0, 1]]);
        assert_eq!(mat1.clone()-col_match, col_result);
    }

    #[test]
    fn la_matrix_addition() {
        let sub = vec![1,2,3,4];
        let mat1 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let mat2 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub]);
        let res0 = vec![2,4,6,8];
        let mat3 = Matrix::<i32>::from_vec2(vec![res0.clone(),res0.clone(),res0.clone()]);
        assert_eq!(mat1.clone() + mat2, mat3);
        let row_match = Matrix::<i32>::from_vec(vec![1,1,1,1]);
        let res = vec![2,3,4,5];
        let row_result = Matrix::<i32>::from_vec2(vec![res.clone(), res.clone(), res.clone()]);
        assert_eq!(mat1.clone() + row_match, row_result);
        let col_match = Matrix::<i32>::from_vec2(vec![vec![1],vec![2],vec![3]]);
        let col_result = Matrix::<i32>::from_vec2(vec![res, vec![3,4,5,6], vec![4,5,6,7]]);
        assert_eq!(mat1.clone()+col_match, col_result);
    }

    #[test]
    fn la_matrix_multiplication() {
        let sub = vec![1,2,3,4];
        let mat1 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let mat2 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let res0 = vec![1,4,9,16];
        let mat3 = Matrix::<i32>::from_vec2(vec![res0.clone(),res0.clone(),res0.clone()]);
        assert_eq!(mat1.clone() * mat2, mat3);
        let row_match = Matrix::<i32>::from_vec(vec![2,2,2,2]);
        let res = vec![2,4,6,8];
        let row_result = Matrix::<i32>::from_vec2(vec![res.clone(), res.clone(), res.clone()]);
        assert_eq!(mat1.clone() * row_match, row_result);
        assert_eq!(mat1.clone() * Matrix::like_with(&mat1, 2), row_result);
        let col_match = Matrix::<i32>::from_vec2(vec![vec![1],vec![2],vec![3]]);
        let col_result = Matrix::<i32>::from_vec2(vec![sub, res, vec![3,6,9,12]]);
        assert_eq!(mat1.clone() * col_match, col_result);
    }


    #[test]
    fn la_matrix_division() {
        let sub = vec![2,4,6,8];
        let mat1 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let mat2 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let res0 = vec![1,1,1,1];
        let mat3 = Matrix::<i32>::from_vec2(vec![res0.clone(),res0.clone(),res0.clone()]);
        assert_eq!(mat1.clone() / mat2, mat3);
        let row_match = Matrix::<i32>::from_vec(vec![2,2,2,2]);
        let res = vec![1,2,3,4];
        let row_result = Matrix::<i32>::from_vec2(vec![res.clone(), res.clone(), res.clone()]);
        assert_eq!(mat1.clone() / row_match, row_result);
        let col_match = Matrix::<i32>::from_vec2(vec![vec![1],vec![2],vec![2]]);
        let col_result = Matrix::<i32>::from_vec2(vec![sub, res.clone(), res]);
        assert_eq!(mat1.clone() / col_match, col_result);
    }

    #[test]
    fn la_matrix_negative() {
        let sub = vec![2,3,4,5];
        let mat1 = Matrix::<i32>::from_vec2(vec![sub.clone(), sub.clone(), sub.clone()]);
        let res = vec![-2, -3, -4, -5];
        let mat2 = Matrix::<i32>::from_vec2(vec![res.clone(), res.clone(), res.clone()]);
        assert_eq!(-mat1, mat2);
        let v = Matrix::from_vec(res);
        let v_res = Matrix::<i32>::from_vec(sub);
        assert_eq!(-v, v_res);
    }

    #[test]
    fn la_matrix_transpose() {
        let mut m = Matrix::from_vec2(vec![vec![1,2], vec![3,4], vec![5,6]]);
        let exp = Matrix::from_vec2(vec![vec![1,3,5], vec![2,4,6]]);
        m.transpose();
        assert_eq!(m, exp);
    }

    #[test]
    fn la_matrix_flatten() {
        let mut m = Matrix::from_vec2(vec![vec![1,2], vec![3,4]]);
        let r = Matrix::from_vec(vec![1,2,3,4]);
        assert_eq!(m.copy_flat(), r);
        m.flatten();
        assert_eq!(m, r);
    }

    #[test]
    fn la_matrix_every() {
        let mut m = Matrix::from_vec2(vec![vec![1,2], vec![3,4]]);
        assert!(m.every(|x| x > 0));
        assert!(!m.every(|x| x < 4));
    }

    #[test]
    fn la_matrix_clip() {
        let m = Matrix::from_vec2(vec![vec![2,3], vec![4,5]]);
        m.clip(3,4);
        let r = Matrix::from_vec2(vec![vec![3,3], vec![4,4]]);
        assert_eq!(m,r);
    }

    #[test]
    fn la_matrix_max() {
        let v = Matrix::from_vec(vec![1,2,3]);
        assert_eq!(v.max(), 3);
        let m = Matrix::from_vec2(vec![vec![2,3], vec![4,5]]);
        assert_eq!(m.max(), 5);
        assert_eq!(m.max_axis(0, false), Matrix::from_vec(vec![4,5]));
        assert_eq!(m.max_axis(0, true), Matrix::from_vec2(vec![vec![4,5]]));
        assert_eq!(m.max_axis(1, false), Matrix::from_vec(vec![3,5]));
        assert_eq!(m.max_axis(1, true), Matrix::from_vec2(vec![vec![3],vec![5]]));
    }

    #[test]
    fn la_matrix_argmax() {
        let v = Matrix::from_vec(vec![1,2,3]);
        assert_eq!(v.argmax(), (0,2));
        let m = Matrix::from_vec2(vec![vec![2,3], vec![4,5]]);
        assert_eq!(m.argmax(), (1, 1));
        assert_eq!(m.argmax_axis(0), Matrix::from_vec(vec![1,1]));
        assert_eq!(m.argmax_axis(1), Matrix::from_vec(vec![1,1]));
    }
    #[test]
    fn la_matrix_sum() {
        let v = Matrix::from_vec(vec![1,2,3]);
        assert_eq!(v.sum(), 6);
        let m = Matrix::from_vec2(vec![vec![2,3], vec![4,5]]);
        assert_eq!(m.sum(), 14);
        assert_eq!(m.sum_axis(0, false), Matrix::from_vec(vec![6,8]));
        assert_eq!(m.sum_axis(0, true), Matrix::from_vec2(vec![vec![6, 8]]));
        assert_eq!(m.sum_axis(1, false), Matrix::from_vec(vec![5,9]));
        assert_eq!(m.sum_axis(1, true), Matrix::from_vec2(vec![vec![5],vec![9]]));
    }


    #[test]
    fn la_matrix_mean() {
        let m = Matrix::from_vec2(vec![vec![2,3], vec![4,5], vec![6,7]]);
        assert_eq!(m.mean(), 4.5);
        assert_eq!(m.mean_axis(0, false), Matrix::from_vec(vec![4.0, 5.0]));
        assert_eq!(m.mean_axis(0, true),Matrix::from_vec2(vec![vec![4.0, 5.0]]));
        assert_eq!(m.mean_axis(1, false), Matrix::from_vec(vec![2.5, 4.5, 6.5]));
        assert_eq!(m.mean_axis(1, true),Matrix::from_vec2(vec![vec![2.5], vec![4.5], vec![6.5]]));
    }
    #[test]
    fn la_matrix_dot() {
        let v1 = Matrix::from_vec(vec![1,2,3,4]);
        let v2 = Matrix::from_vec(vec![5,6,7,8]);
        assert_eq!(v1.v_dot(&v2), 70);
        let m1 = Matrix::from_vec2(vec![vec![1,2,3,4], vec![5,6,7,8], vec![9,10,11,12]]);
        let m2 = Matrix::from_vec2(vec![vec![1,2,3,4], vec![5,6,7,8], vec![9,10,11,12], vec![13,14,15,16]]);
        let mat_mul_res = Matrix::from_vec2(vec![vec![90, 100, 110, 120], vec![202, 228, 254, 280], vec![314, 356, 398, 440]]);
        assert_eq!(m1.mat_mul(&m2), mat_mul_res);
        assert_eq!(m1.dot(&m2), mat_mul_res);
        let mat_vec_res = Matrix::from_vec(vec![30, 70, 110]);
        assert_eq!(m1.dot(&v1), mat_vec_res);
    }

    #[test]
    fn la_matrix_ln() {
        let m = Matrix::from_vec2(vec![vec![2.0,3.0], vec![4.0,5.0]]);
        let r = Matrix::from_vec2(vec![
 	            vec![0.6931471805599453, 1.0986122886681098],
 	            vec![1.3862943611198906, 1.6094379124341003]]);
        assert_eq!(m.ln_copy(), r);
        let neg_r = -(r.clone());
        assert_eq!(m.neg_ln_copy(), neg_r);
        m.neg_ln();
        assert_eq!(m, -r);
    }

    #[test]
    fn la_matrix_exp() {
        let m = Matrix::from_vec2(vec![vec![2.0,3.0], vec![4.0,5.0]]);
        let r = Matrix::from_vec2(vec![
            vec![7.3890560989306495, 20.085536923187664],
            vec![54.59815003314423, 148.41315910257657]]);
        m.exp();
        assert_eq!(m, r);
    }

    #[test]
    fn la_matrix_element_iter() {
        let mat = Matrix::from_vec2(vec![vec![1,2,3,4,5], vec![6,7,8,9,10]]);
        let res: Vec<i32> = (1..11).into_iter().collect();
        let mut res_iter = res.into_iter();
        for element in mat {
            assert_eq!(element, res_iter.next().unwrap());
        }

    }

    #[test]
    fn la_matrix_reshape() {
        let mut mat = Matrix::from_vec2(vec![vec![1,2], vec![3,4], vec![5,6]]);
        let res_mat = Matrix::from_vec2(vec![vec![1,2,3], vec![4,5,6]]);
        let mut mat_c = mat.clone();
        mat_c.reshape((2,3));
        assert_eq!(mat_c, res_mat);
        mat_c.reshape((-1, 2));
        assert_eq!(mat_c, mat);
        let flat = Matrix::from_vec2(vec![vec![1,2,3,4,5,6]]);
        mat_c.reshape((1,-1));
        assert_eq!(mat_c, flat);
        let mut v = Matrix::from_vec(vec![0.7, 0.1, 0.2]);
        v.reshape((-1,1));
        let res = Matrix::from_vec2(vec![vec![0.7], vec![0.1], vec![0.2]]);
        assert_eq!(v, res);
    }
    
}  
