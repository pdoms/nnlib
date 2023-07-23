use core::fmt;
use std::alloc;
use std::f32::{NEG_INFINITY, consts::E};
use std::ops::{Range, Sub, Add, Mul, Div, Neg};
use rand::{self, Rng};



pub struct Matrix {
    pub cols: usize,
    pub rows: usize,
    pub stride: usize,
    pub ptr: *mut f32,
    pub len: usize,
}

type MatIdx = (usize, usize);


macro_rules! mat {
    ($rows:expr, $cols:expr) => {
        Matrix::zeroed($rows, $cols)
    }
}

macro_rules! mat_dump {
    ($mat:ident) => {
        {
            if $mat.is_vector() {
                let mut s = String::from("[ ");
                for i in 0..$mat.len {
                    s = format!("{s} {}", $mat.get((0, i)));
                }
                s = format!("{s} ]\n");
                println!("{s}");
                s
            } else {
                let mut s = String::from("[\n");
                for i in 0..$mat.rows {
                    s = format!("{s}\t[");
                    for j in 0..$mat.cols {
                        s = format!("{s} {}", $mat.get((i, j)));
                    }
                    s = format!("{s} ]\n");
                }
                s = format!("{s}]\n");
                println!("{s}");
                s
            }
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(mat_dump!(self).as_str())
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(mat_dump!(self).as_str())
    }
}


impl PartialEq for Matrix {
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

impl Eq for Matrix {}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            let mem_size =std::mem::size_of::<f32>(); 
            let layout = alloc::Layout::from_size_align_unchecked(                  
                mem_size*self.len,
                std::mem::align_of::<f32>());
            alloc::dealloc(self.ptr as *mut u8, layout)
        }
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mem_size =std::mem::size_of::<f32>(); 
        let layout = alloc::Layout::array::<f32>(self.len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc(layout)
        } as *mut f32;
        unsafe {
            std::ptr::copy(self.ptr, ptr, self.len*mem_size)
        };
        Self {
            cols: self.cols,
            rows: self.rows,
            stride: self.stride,
            len: self.len,
            ptr
        }
    }
}

impl Sub for Matrix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.subtract(&rhs)
    }
}

impl Add for Matrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.addition(&rhs)
    }
}

impl Mul for Matrix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.multiplication(&rhs)
    }
}

impl Div for Matrix {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.division(&rhs)
    }
}

impl Neg for Matrix {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.negative();
        return self
    }
}

impl Matrix {
    /// rows, cols
    pub fn new(rows: usize, cols: usize) -> Self {
        let mem_size =std::mem::size_of::<f32>(); 
        let mut len = cols*rows;
        if rows == 0 {
            len = cols*1;
        }
        let layout = alloc::Layout::array::<f32>(len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc(layout)
        } as *mut f32;
        Self {
            cols,
            rows,
            stride: cols,
            ptr,
            len
        }
    }

    pub fn zeroed(rows: usize, cols: usize) -> Self {
        let mem_size =std::mem::size_of::<f32>(); 
        let mut len = cols*rows;
        if rows == 0 {
            len = cols*1;
        }
        let layout = alloc::Layout::array::<f32>(len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc_zeroed(layout)
        } as *mut f32;
        Self {
            cols,
            rows,
            stride: cols,
            ptr,
            len
        }
    }

    pub fn random(rows: usize, cols: usize,  range: Range<f32>) -> Self {
        let matrix = Self::zeroed(rows, cols);
        let mut rng = rand::thread_rng();
        unsafe {
            for i in 0..matrix.len {
                let r: f32 = rng.gen_range(range.clone());
                matrix.ptr.add(i).write(r);
            }
        }
        matrix
    }

    pub fn from_vec(mut data: Vec<f32>) -> Self {
        let len = data.len();
        let mem_size =std::mem::size_of::<f32>(); 
        let layout = alloc::Layout::array::<f32>(len*mem_size).expect("Could not create layout");
        let ptr = unsafe {
            alloc::alloc(layout)
        } as *mut f32;
        unsafe {ptr.copy_from(data.as_mut_ptr(), len)};
        Self {
            cols: len,
            rows: 0,
            stride: 1,
            ptr,
            len
        }

    }
    pub fn from_vec2(data: Vec<Vec<f32>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        let matrix = Self::zeroed(rows, cols);
        let mut flattened: Vec<f32> = data.into_iter().flatten().collect();
        unsafe {
            matrix.ptr.copy_from(flattened.as_mut_ptr(), matrix.len);
        }
        matrix
    }

    pub fn shape(&self) -> MatIdx {
        (self.rows, self.cols)
    }

    pub fn get(&self, index: MatIdx) -> f32 {
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

    pub fn set(&self, f: f32, index: MatIdx) {
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
    fn get_row(&self, row: usize) -> Matrix {
        if self.is_vector() {
            return self.clone()
        }
        assert!(row < self.rows, "index out of range");
        let mut vector = Vec::<f32>::with_capacity(self.cols);
        unsafe {
            let offset = self.ptr.add(row*self.stride);
            for i in 0..self.cols {
                vector.push(offset.add(i).read())
            }
        };
        Matrix::from_vec(vector)
    }

    fn set_row(&self, i: usize, row: &Matrix) {
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

    fn set_col(&self, i: usize, col: &Matrix) {
        assert!(col.is_vector(), "src is not a vector");
        assert!(self.rows == col.cols, "rows don't match");
        self.bound_check_cols(i);
        for j in 0..self.rows {
            let v = col.get((0, j));
            self.set(v, (j, i));
        }
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.shape_match(other) {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((i, j));
                    result.set(lhs - rhs, (i, j))
                }
            }
            result
        } else if self.col_match(other) && other.is_vector() {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((0, j));
                    result.set(lhs - rhs, (i, j))
                }
            }
            result
        } else if self.row_match(other) && other.cols == 1 {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.cols {
                let col = self.get_col(i);
                for j in 0..self.rows {
                    let lhs = col.get((0,j));
                    let rhs = other.get((j, 0));
                    col.set(lhs - rhs, (0, j));
                }
                result.set_col(i, &col)
            }
            result
        } else {
            panic!("cannot match minuend and subtrahend")
        }

    }

    pub fn addition(&self, other: &Matrix) -> Matrix {
        if self.shape_match(other) {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((i, j));
                    result.set(lhs + rhs, (i, j))
                }
            }
            result
        } else if self.col_match(other) && other.is_vector() {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((0, j));
                    result.set(lhs + rhs, (i, j))
                }
            }
            result
        } else if self.row_match(other) && other.cols == 1 {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.cols {
                let col = self.get_col(i);
                for j in 0..self.rows {
                    let lhs = col.get((0,j));
                    let rhs = other.get((j, 0));
                    col.set(lhs + rhs, (0, j));
                }
                result.set_col(i, &col)
            }
            result
        } else {
            panic!("cannot match addends")
        }
    }


    pub fn multiplication(&self, other: &Matrix) -> Matrix {
        if self.shape_match(other) {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((i, j));
                    result.set(lhs * rhs, (i, j))
                }
            }
            result
        } else if self.col_match(other) && other.is_vector() {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((0, j));
                    result.set(lhs * rhs, (i, j))
                }
            }
            result
        } else if self.row_match(other) && other.cols == 1 {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.cols {
                let col = self.get_col(i);
                for j in 0..self.rows {
                    let lhs = col.get((0,j));
                    let rhs = other.get((j, 0));
                    col.set(lhs * rhs, (0, j));
                }
                result.set_col(i, &col)
            }
            result
        } else {
            panic!("cannot match multiplicand and multiplier")
        }
    }

    pub fn division(&self, other: &Matrix) -> Matrix {
        if self.shape_match(other) {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((i, j));
                    result.set(lhs / rhs, (i, j))
                }
            }
            result
        } else if self.col_match(other) && other.is_vector() {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let lhs = self.get((i,j));
                    let rhs = other.get((0, j));
                    result.set(lhs / rhs, (i, j))
                }
            }
            result
        } else if self.row_match(other) && other.cols == 1 {
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.cols {
                let col = self.get_col(i);
                for j in 0..self.rows {
                    let lhs = col.get((0,j));
                    let rhs = other.get((j, 0));
                    col.set(lhs / rhs, (0, j));
                }
                result.set_col(i, &col)
            }
            result
        } else {
            panic!("cannot match dividend and divisor")
        }
    }

    pub fn negative(&self) {
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0,j));
                self.set(-v, (0, j))
            }
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i,j));
                    self.set(-v, (i, j))
                }
            }
        }
    }



    fn get_col(&self, col: usize) -> Matrix {
        assert!(col < self.cols, "index out of range");
        let mut vector = Vec::<f32>::with_capacity(self.rows);
        let mut idx = col;
        for _ in 0..self.rows {
            let val = unsafe {
                self.ptr.add(idx).read()
            };
            vector.push(val);
            idx += self.stride;
        }
        Matrix::from_vec(vector)
    }

    pub fn transpose(&self) {
        //new matrix
        //get column of self
        //set it as row in new matrix
        //switch self.cols with self.rows
        //copy new matrix data to self.ptr data
    }

    pub fn scalar_mul(&self, scalar: f32) {
        unsafe {
            for i in 0..self.len {
                let offset = self.ptr.add(i);
                let mut value = offset.read();
                value *= scalar;
                offset.write(value);
            }
        }
    }



    pub fn scalar_div(&self, scalar: f32) {
        if scalar == 0.0 {
            panic!("Division by zero impossible");
        }
        unsafe {
            for i in 0..self.len {
                let offset = self.ptr.add(i);
                let mut value = offset.read();
                value /= scalar;
                offset.write(value);
            }
        }
    }

    pub fn mmul(&self, other: &Matrix) ->Matrix {
        if self.can_multipy(&other) {
            let trg = Matrix::new(self.rows, other.cols);
            for i in 0..self.rows {
                let v1 = self.get_row(i);
                for j in 0..other.cols {
                    let v2 = other.get_col(j);
                    let dot = v1.vector_dot(&v2);
                    trg.set(dot, (i, j));
                }
            }
            trg
        } else {
            panic!("M1 cols ({}) don't match rows ({})", self.cols, other.rows)
        }
    }

    pub fn vector_dot(&self, other: &Matrix) -> f32 {
        assert!(self.is_vector() && other.is_vector() && self.cols == other.cols);
        let mut result: f32 = 0.0;
        unsafe {
            for i in 0..self.cols {
                let v1 = self.ptr.add(i).read();    
                let v2 = other.ptr.add(i).read();
                result += v1 * v2;
            }
        }
        result
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        if !self.is_vector() && !other.is_vector() {
            //matrix multiplication
            self.mmul(other)
        } else {
            //"normal" dot product
            assert!(other.is_vector() && self.cols == other.cols, "M1 cols don't match rows ");
            let result = Matrix::new(0, self.rows);
            for i in 0..self.rows {
                let row = self.get_row(i);
                result.set(row.vector_dot(other), (0, i))
            }
            result
        }
    }

    pub fn add_vector(&self, other: &Matrix) {
        assert!(self.cols == other.cols, "cols don't match");
        assert!(other.is_vector(), "other is not a vector");
        for i in 0..self.rows {
            for j in 0..self.cols {
                let v1 = self.get((i, j));
                let v2 = other.get((0, j));
                self.set(v1+v2, (i, j));
            }
        }
        
    }

    pub fn vector_add(&self, other: &Matrix) -> Matrix {
        assert!(self.is_vector() && other.is_vector() && self.cols == other.cols);
        let mut result: Vec<f32> = Vec::with_capacity(self.cols);
        unsafe {
            for i in 0..self.cols {
                let v1 = self.ptr.add(i).read();    
                let v2 = other.ptr.add(i).read();
                result.push(v1 + v2);
            }
        }
        Matrix::from_vec(result) 
    }

    ///axis0
    ///| | |   |
    ///| V |   |
    ///|   |   |
    
    ///axis1
    ///|   |   |
    ///|->  -> |
    ///|   |   |
    
    pub fn sum(&self) -> f32 {
        let mut acc = 0.0;
        unsafe {
            for i in 0..self.len {
                acc += self.ptr.add(i).read();    
            }
        };
        acc
    }

    pub fn sum_axis(&self, axis: usize, keepdims: bool) -> Matrix {
        match axis {
           0 => {
                let mut result = Matrix::new(self.cols, 1);
                for i in 0..self.cols {
                    let col = self.get_col(i);
                    result.set(col.sum(), (i, 0));
                }
                if keepdims {
                    result
                } else {
                    result.flatten();
                    result
                }
            },
            1 => {
                let mut result = Matrix::new(self.rows, 1);
                for i in 0..self.rows {
                    let col = self.get_row(i);
                    result.set(col.sum(), (i, 0));
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

    pub fn max(&self) -> f32 {
        let mut max = NEG_INFINITY;
        unsafe {
            for i in 0..self.len {
                let v = self.ptr.add(i).read();    
                if v > max {
                    max = v
                }
            }
        };
        if max == NEG_INFINITY {
            panic!("could not determine max")
        }
        max
    }
    pub fn max_axis(&self, axis: usize, keepdims: bool) -> Matrix {
        match axis {
           0 => {
                let mut result = Matrix::new(1, self.cols);
                for i in 0..self.cols {
                    let col = self.get_col(i);
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
                let mut result = Matrix::new(self.rows, 1);
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

    pub fn log(&self) -> Matrix {
        self.for_each_set(|x| x.ln())
    }

    pub fn neg_log(&self) -> Matrix {
        self.for_each_set(|x| -(x.ln()))
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.len as f32
    }
    pub fn mean_axis(&self,axis: usize, keepdims: bool) -> Matrix {
        match axis {
           0 => {
                let mut result = Matrix::new(0, self.cols);
                    for i in 0..self.cols {
                        let col = self.get_col(i);
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
                let mut result = Matrix::new(self.rows, 1);
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

    pub fn flatten(&mut self) {
        self.cols = self.len;
        self.rows = 0;
        self.stride = 1;
    }

    pub fn maximum_scalar(&self, s: f32) -> Matrix {
        let result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            let row = self.get_row(i);
            let maxes = row.vector_scalar_maximum(s);
            result.set_row(i, &maxes);
        }
        result
    }
    pub fn maximum(&self, other: &Matrix) -> Matrix {
        if other.is_vector() {
            assert!(self.cols == other.cols, "maximum() requires matching number of columns");
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                let row = self.get_row(i);
                let maxes = row.vector_maximum(other);
                result.set_row(i, &maxes);
            }
            result

        } else {
            assert!(self.shape_match(other), "maximum() requires shapes to match of both items");
            let result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                let row_self = self.get_row(i);
                let row_other = other.get_row(i);
                let maxes = row_self.vector_maximum(&row_other);
                result.set_row(i, &maxes);
            }
            result
        }
    }

    // scalar compared with each element in vector, returns new matr
    pub fn vector_scalar_maximum(&self, s: f32) -> Matrix {
        assert!(self.is_vector(), "vector_scalar_maximum() requires self to be a vector");
        let result = Matrix::new(0, self.len);
        for i in 0..self.len {
            let a = unsafe {self.ptr.add(i).read()};
            let max = max(a, s);
            result.set(max, (0, i));
        }
        result
    }
    // vector with vector returns new vector, with maxes 
    pub fn vector_maximum(&self, other: &Matrix) -> Matrix {
        assert!(self.is_vector(), "vector_maximum() requires self to be a vector");
        assert!(other.is_vector(), "vector_maximum() requires other to be a vector");
        assert!(self.len == other.len, "vector_maximum() requires self and other to be of the same length");
        let result = Matrix::new(0, self.len);
        for i in 0..self.len {
            let a = unsafe {self.ptr.add(i).read()};
            let b = unsafe {other.ptr.add(i).read()};
            let max = max(a, b);
            result.set(max, (0,i));
        }
        result
    }

    pub fn exp(&self) -> Matrix {
        self.for_each_set(|x| {E.powf(x)})
    }

    pub fn all<F>(&self, f: F) -> bool 
    where 
        F: Fn(f32) -> bool  
    {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let v = self.get((i,j));
                if !f(v) {
                    println!("received 'false' for '{v}'");
                    return false;
                }
            }
        }
        true

    }

    pub fn for_each_set<F>(&self, f: F) -> Matrix 
    where 
        F: Fn(f32) -> f32  
    {
        let result = Matrix::new(self.rows, self.cols);
        if self.is_vector() {
            for j in 0..self.cols {
                let v = self.get((0,j));
                let adjusted = f(v);
                result.set(adjusted, (0,j))
            }
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get((i,j));
                    let adjusted = f(v);
                    result.set(adjusted, (i,j))
                }
            }
        }
        result
    } 

    pub fn clip(&self, low: f32, high: f32) {
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

    pub fn is_vector(&self) -> bool {
        self.rows == 0  
    }
    fn can_multipy(&self, other: &Matrix) -> bool {
        self.cols == other.rows
    }
    fn shape_match(&self, other: &Matrix) -> bool {
        self.cols == other.cols && self.rows == other.rows
    }
    fn col_match(&self, other: &Matrix) -> bool {
        self.cols == other.cols
    }
    fn row_match(&self, other: &Matrix) -> bool {
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




fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a >= b {
        a
    } else {
        b
    }
}





#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn la_matrix_struct() {
        let mat_1 = Matrix::zeroed(2, 2);
        assert_eq!(mat_1.get((0,0)), 0.0);
        assert_eq!(mat_dump!(mat_1), "[\n\t[ 0 0 ]\n\t[ 0 0 ]\n]\n");
        let mat_2 = Matrix::random(2, 2, -0.1..0.1);
        let val1 = mat_2.get((0,0));
        let val2 = mat_2.get((1,1));
        assert!(val1 > -1.0 && val1 < 1.0);
        assert!(val2 > -1.0 && val1 < 1.0);
        let v1 = vec![1.0,2.0,3.0,4.0,5.0,6.0];
        let v2 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mat_3 = Matrix::from_vec(v1.clone());
        let mat_4 = Matrix::from_vec2(v2.clone());
        assert_eq!(mat_3.cols, v1.len());
        assert_eq!(mat_3.get((0,1)), 2.0);
        assert_eq!(mat_4.rows, v2.len());
        assert_eq!(mat_4.cols, v2[0].len());
        assert_eq!(mat_4.get((1,1)), 4.0);
        let mat_5 = mat!(2,2);
        let mat_6 = mat!(2,2);
        assert_eq!(mat_5, mat_6);
        let mat_7 = Matrix::from_vec(vec![3.0, 4.0]);
        assert_eq!(mat_4.get_row(1), mat_7);
        let mat_8 = Matrix::from_vec2(v2);
        let col = Matrix::from_vec(vec![2.0, 4.0, 6.0]);
        assert_eq!(mat_8.get_col(1), col);
    }

    #[test]
    fn la_matrix_set() {
        let mat = mat!(4,3);
        assert_eq!(mat.get((1,1)), 0.0);
        mat.set(4.0, (1,1));
        assert_eq!(mat.get((1,1)), 4.0);
        let row = Matrix::from_vec(vec![1.0, 2.0, 3.0]);
        mat.set_row(3, &row);
        assert_eq!(mat.get_row(3), row);
        let col = Matrix::from_vec(vec![2.0, 3.0]);
        let mat1 = Matrix::zeroed(2, 3);
        let exp = Matrix::from_vec2(vec![vec![0.0, 2.0, 0.0], vec![0.0, 3.0, 0.0]]);
        mat1.set_col(1, &col);
        assert_eq!(exp, mat1);
    }

    #[test]
    fn la_matrix_scalar_mul() {
        let v1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mat1 = Matrix::from_vec2(v1);
        let v2 = vec![vec![2.0, 4.0], vec![6.0, 8.0], vec![10.0, 12.0]];
        let mat2 = Matrix::from_vec2(v2);
        mat1.scalar_mul(2.0);
        assert_eq!(mat1, mat2);
    }

    #[test]
    fn la_matrix_scalar_div() {
        let v1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let v2 = vec![vec![2.0, 4.0], vec![6.0, 8.0], vec![10.0, 12.0]];
        let mat1 = Matrix::from_vec2(v1);
        let mat2 = Matrix::from_vec2(v2);
        mat2.scalar_div(2.0);
        assert_eq!(mat1, mat2)
    }

    #[test]
    fn la_matrix_mul_dot() {
        let m1 = Matrix::from_vec2(vec![vec![1.0,2.0,3.0], vec![4.0,5.0,6.0]]);
        let m2 = Matrix::from_vec2(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0,12.0]]);
        let result = Matrix::from_vec2(vec![vec![58.0, 64.0], vec![139.0, 154.0]]);
        assert_eq!(m1.mmul(&m2), result);
        let mat_4 = Matrix::from_vec2(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let v1 = Matrix::from_vec(vec![5.0, 6.0]);
        let exptected = Matrix::from_vec(vec![17.0, 39.0]);
        assert_eq!(mat_4.dot(&v1), exptected);
    }

    #[test]
    fn la_matrix_sums() {
        let mat = Matrix::from_vec2(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(mat.sum(), 10.0);

        let mat1 = Matrix::from_vec2(vec![vec![0.0,1.0], vec![0.0, 5.0]]);
        let exp = Matrix::from_vec2(vec![vec![0.0], vec![6.0]]);
        let res = mat1.sum_axis(0, true);
        assert_eq!(res, exp);
        let exp_flat = Matrix::from_vec(vec![0.0, 6.0]);
        let res = mat1.sum_axis(0, false);
        assert_eq!(res, exp_flat);
        
        let exp2 = Matrix::from_vec2(vec![vec![1.0], vec![5.0]]);
        let res1 = mat1.sum_axis(1, true);
        assert_eq!(res1, exp2);

        let exp_flat = Matrix::from_vec(vec![1.0, 5.0]);
        let res = mat1.sum_axis(1, false);
        assert_eq!(res, exp_flat);
        
    }

    #[test]
    fn la_matrix_max() {
        let mat = Matrix::from_vec2(vec![vec![0.0,1.0], vec![2.0,3.0]]);
        let max = mat.max();
        let exp = 3.0;
        assert_eq!(exp, max);

        let max_0 = mat.max_axis(0, true);
        let exp_0 = Matrix::from_vec2(vec![vec![2.0, 3.0]]);
        assert_eq!(exp_0, max_0);

        let max_1 = mat.max_axis(1, true);
        let exp_1 = Matrix::from_vec2(vec![vec![1.0], vec![3.0]]);
        assert_eq!(exp_1, max_1);

    }

    #[test]
    fn la_matrix_mean() {
        let mat = Matrix::from_vec2(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let exp_all = 2.5;
        let exp_0 = Matrix::from_vec(vec![2.0, 3.0]);
        let exp_1 = Matrix::from_vec(vec![1.5, 3.5]);
        assert_eq!(mat.mean(), exp_all);
        assert_eq!(mat.mean_axis(0, false), exp_0);
        assert_eq!(mat.mean_axis(1, false), exp_1);
    }

    #[test]
    fn la_matrix_maximum() {
        let mat5 = Matrix::from_vec2(vec![vec![-1.0, 0.0, 1.0], vec![-0.5, 0.1, 0.5]]);
        let mat6 = Matrix::from_vec2(vec![vec![-0.9, 0.001, 0.9], vec![-0.4, 0.01, 0.49]]);
        let vec_5 = Matrix::from_vec(vec![-0.5, 0.1, 0.9]);
        let res_5 = Matrix::from_vec2(vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.1, 0.5]]);
        let res_6 = Matrix::from_vec2(vec![vec![-0.9, 0.001, 1.0], vec![-0.4, 0.1, 0.5]]);
        let res_7 = Matrix::from_vec2(vec![vec![-0.5, 0.1, 1.0], vec![-0.5, 0.1, 0.9]]);
        assert_eq!(mat5.maximum_scalar(0.0), res_5);
        assert_eq!(mat5.maximum(&mat6), res_6);
        assert_eq!(mat5.maximum(&vec_5), res_7);
    }   

    #[test]
    fn la_matrix_subtract() {
        let mat = Matrix::from_vec2(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0]]);
        let row = Matrix::from_vec(vec![0.0, 1.0, 2.0]);
        let exp_row = Matrix::from_vec2(vec![vec![0.0, 0.0, 0.0], vec![3.0, 3.0, 3.0], vec![6.0, 6.0, 6.0]]);
        let res_row = mat.subtract(&row);
        assert_eq!(exp_row, res_row);
        let col = Matrix::from_vec2(vec![vec![0.0], vec![1.0], vec![2.0]]);
        let exp_col = Matrix::from_vec2(vec![vec![0.0, 1.0, 2.0], vec![2.0, 3.0, 4.0], vec![4.0, 5.0, 6.0]]);
        let res_col = mat.clone() - col;
        assert_eq!(exp_col, res_col);
        let mat2 = Matrix::from_vec2(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0]]);
        let exp_mat2 = Matrix::zeroed(3,3); 
        let res_mat2 = mat.clone() - mat2;
        assert_eq!(exp_mat2, res_mat2);
    }

    #[test]
    fn la_matrix_addition() {
        let mat = Matrix::from_vec2(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0]]);
        let row = Matrix::from_vec(vec![0.0, 1.0, 2.0]);
        let exp_row = Matrix::from_vec2(vec![vec![0.0, 2.0, 4.0], vec![3.0, 5.0, 7.0], vec![6.0, 8.0, 10.0]]);
        let res_row = mat.clone().addition(&row);
        assert_eq!(exp_row, res_row);
        let col = Matrix::from_vec2(vec![vec![0.0], vec![1.0], vec![2.0]]);
        let exp_col = Matrix::from_vec2(vec![vec![0.0, 1.0, 2.0], vec![4.0, 5.0, 6.0], vec![8.0, 9.0, 10.0]]);
        let res_col = mat.clone() + col;
        assert_eq!(exp_col, res_col);
        let mat2 = Matrix::from_vec2(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0]]);
        let exp_mat2 = Matrix::from_vec2(vec![vec![0.0, 2.0, 4.0], vec![6.0, 8.0, 10.0], vec![12.0, 14.0, 16.0]]); 
        let res_mat2 = mat.clone() + mat2;
        assert_eq!(exp_mat2, res_mat2);
    }

    #[test]
    fn la_matrix_multiplication() {
        let mat = Matrix::from_vec2(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0]]);
        let row = Matrix::from_vec(vec![0.0, 1.0, 2.0]);
        let exp_row = Matrix::from_vec2(vec![vec![0.0, 1.0, 4.0], vec![0.0, 4.0, 10.0], vec![0.0, 7.0, 16.0]]);
        let res_row = mat * row;
        assert_eq!(exp_row, res_row);
    }

    #[test]
    fn la_matrix_division() {
        let mat = Matrix::from_vec2(vec![vec![3.0, 4.0, 5.0], vec![6.0, 8.0, 10.0]]);
        let col = Matrix::from_vec2(vec![vec![1.0], vec![2.0]]);
        let exp_col = Matrix::from_vec2(vec![vec![3.0, 4.0, 5.0], vec![3.0, 4.0, 5.0]]);
        let res_col = mat / col;
        assert_eq!(exp_col, res_col);
    }

    #[test]
    fn la_matrix_for_each() {
        let mat5 = Matrix::from_vec2(vec![vec![-1.0, 0.0, 1.0], vec![-0.5, 0.1, 0.5]]);
        let res_8 = mat5.for_each_set(|x| {if x >= 0.0 {return x} else {
            return 0.01*x
        }});
        let exp_8 = Matrix::from_vec2(vec![vec![-0.01, 0.0, 1.0], vec![-0.005, 0.1, 0.5]]);
        assert_eq!(res_8, exp_8);
    }


    #[test]
    fn la_matrix_exponent() {
        let mat_exp = Matrix::from_vec2(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let exp_exp = Matrix::from_vec2(vec![
            vec![2.7182817, 7.3890557],
            vec![20.085535, 54.598145]
        ]);
        let res = mat_exp.exp();
        assert_eq!(res, exp_exp);
    }

    #[test]
    fn la_matrix_clip() {
        let data = Matrix::from_vec2(vec![vec![0.0, 1.1], vec![0.01, 0.5]]);
        let exp = Matrix::from_vec2(vec![vec![0.0001, 0.99], vec![0.01, 0.5]]);
        data.clip(0.0001, 0.99);
        assert_eq!(data, exp);
    }

    #[test]
    fn la_matrix_log() {
        let data = Matrix::from_vec(vec![0.7,0.5,0.9]);
        let res_pos = Matrix::from_vec(vec![-0.35667497, -0.6931472, -0.105360545]);
        let res_neg = Matrix::from_vec(vec![0.35667497, 0.6931472, 0.105360545]);
        assert_eq!(data.log(), res_pos);
        assert_eq!(data.neg_log(), res_neg);
        assert_eq!(-data.log(), res_neg);
    }

    #[test]
    fn la_vector_operations() {
        let v1 = Matrix::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = Matrix::from_vec(vec![1.0, 2.0, 3.0]);
        let result = v1.vector_dot(&v2);
        assert_eq!(result, 14.0);
        let sum_result = Matrix::from_vec(vec![2.0, 4.0, 6.0]);
        assert_eq!(v1.vector_add(&v2), sum_result);
        let v3 = Matrix::from_vec(vec![0.0, -1.23, 0.01, 0.02, -1.0]);
        let result2 = v3.vector_scalar_maximum(0.0);
        let exp = Matrix::from_vec(vec![0.0, 0.0, 0.01, 0.02, 0.0]);
        assert_eq!(result2, exp);
        let v4 = Matrix::from_vec(vec![0.000001, -1.22, 0.001, 0.22, 1.0]);
        let exp2 = Matrix::from_vec(vec![0.000001, -1.22, 0.01, 0.22, 1.0]);
        assert_eq!(v3.vector_maximum(&v4), exp2);
    }

    #[test]
    fn la_all_fn() {
        let mat_pos = Matrix::from_vec2(vec![vec![3.333334, 3.34], vec![3.334, 3.333333333334]]);
        let mat_neg = Matrix::from_vec2(vec![vec![3.333334, 3.00], vec![3.334, 3.333333333334]]);
        assert!(mat_pos.all(|x| 3.3334 - x < 0.01));
        assert!(!mat_neg.all(|x| 3.3334 - x < 0.01));
    }
}
