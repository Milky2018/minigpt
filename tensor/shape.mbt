///|
fn shape_size(shape : Array[Int]) -> Int {
  let mut size = 1
  for dim in shape {
    if dim <= 0 {
      abort("tensor dimensions must be positive")
    }
    size *= dim
  }
  size
}

///|
fn copy_ints(xs : Array[Int]) -> Array[Int] {
  xs.copy()
}

///|
fn copy_doubles(xs : Array[Double]) -> Array[Double] {
  xs.copy()
}

///|
fn broadcast_shape(a : Array[Int], b : Array[Int]) -> Array[Int] {
  let rank_a = a.length()
  let rank_b = b.length()
  let rank = if rank_a > rank_b { rank_a } else { rank_b }
  let out = Array::make(rank, 1)
  for i in 0..<rank {
    let ai = rank_a - rank + i
    let bi = rank_b - rank + i
    let da = if ai < 0 { 1 } else { a[ai] }
    let db = if bi < 0 { 1 } else { b[bi] }
    if da == db || da == 1 || db == 1 {
      out[i] = if da > db { da } else { db }
    } else {
      abort("tensor shapes are not broadcast-compatible")
    }
  }
  out
}

///|
fn unravel_index(linear : Int, shape : Array[Int]) -> Array[Int] {
  let mut remaining = linear
  let idx = Array::make(shape.length(), 0)
  let mut axis = shape.length() - 1
  while axis >= 0 {
    let dim = shape[axis]
    idx[axis] = remaining % dim
    remaining = remaining / dim
    if axis == 0 {
      break
    }
    axis -= 1
  }
  idx
}

///|
fn ravel_index(idx : Array[Int], shape : Array[Int]) -> Int {
  let mut linear = 0
  for axis in 0..<shape.length() {
    linear = linear * shape[axis] + idx[axis]
  }
  linear
}

///|
fn broadcast_source_index(out_idx : Array[Int], in_shape : Array[Int]) -> Int {
  if in_shape.length() == 0 {
    0
  } else {
    let offset = out_idx.length() - in_shape.length()
    let in_idx = Array::make(in_shape.length(), 0)
    for axis in 0..<in_shape.length() {
      let dim = in_shape[axis]
      in_idx[axis] = if dim == 1 { 0 } else { out_idx[offset + axis] }
    }
    ravel_index(in_idx, in_shape)
  }
}

///|
fn reduce_to_shape(
  grad : Array[Double],
  from_shape : Array[Int],
  to_shape : Array[Int],
) -> Array[Double] {
  let out = Array::make(shape_size(to_shape), 0.0)
  for i in 0..<grad.length() {
    let out_idx = unravel_index(i, from_shape)
    let src = broadcast_source_index(out_idx, to_shape)
    out[src] += grad[i]
  }
  out
}
