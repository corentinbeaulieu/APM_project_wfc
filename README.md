# Wave Function Collapse to solve sudoku

## Introduction

The main goal of this Master's project is to parallelize a sudoku solver
using OpenMP then to port on GPU using either NVidia CUDA or OpenMP target.

### Wave Function Collapse

### Sudoku solving

## Build 

```sh
$ cmake -B build
$ cmake --build build
```

The executable is located in `build` and is named `wfc`.

## Usage

```
wfc [-h] [-o folder/] [-l solver] [-p count] [-s seeds...] <path/to/file.data>
  -h          print this help message.
  -o          output folder to save solutions. adds the seed to the data file name.
  -p count    number of seeds that can be processed in parallel
  -s seeds    seeds to use. can an integer or a range: `from-to`.
  -l solver   solver to use. possible values are: [cpu], omp, target
```

## Input Format

There two types of possible inputs
<details>
<summary> A simple region or block (a 3x3 elements block in the original game) </summary>

```
s5
0, 0, 0, 1=4
0, 0, 2, 3=1
```

represents the block

|_|4|_|_|_|
---|---|---|---|---
|_|_|_|_|_|
|_|_|_|1|_|
|_|_|_|_|_|
|_|_|_|_|_|

</details>
<details>
<summary> A grid (3x3 blocks in the original game) </summary>

```
g3
0, 0, 0, 1=4
0, 0, 2, 2=1
1, 1, 0, 1=4
2, 2, 2, 2=1
```
represents the grid 

|_|4|_|_|_|_|_|_|_|
---|---|---|---|---|---|---|---|---
|_|_|_|_|_|_|_|_|_|
|_|_|1|_|_|_|_|_|_|
|_|_|_|_|4|_|_|_|_|
|_|_|_|_|_|_|_|_|_|
|_|_|_|_|_|_|_|_|_|
|_|_|_|_|_|_|_|_|_|
|_|_|_|_|_|_|_|_|_|
|_|_|_|_|_|_|_|_|1|

</details>


The format is
- The type (`s` or `g`) followed by the size of the grids on the first line
- The coordinates of an element (grid x, grid y, x, y) followed by `=` and its value

All values have to be between one and the size of the grids.


## Output Format

According to the input format, the program gives two different types of output

<details>
<summary> A simple region or block </summary>

```
grid:  1
block: 2
1 2 3 4
```

represents the block

|1|2|
---|---
|3|4|

</details>
<details>
<summary> A grid </summary>

```
grid:  2
block: 2
1 2 3 4
3 4 1 2
2 1 4 3
4 3 2 1
```

represents the grid 

|1|2|3|4|
---|---|---|---
|3|4|1|2|
|2|1|4|3|
|4|3|2|1|

</details>

The output is by default printed on the standard output. 
Alternatively, you can specify a folder in which the solutions will be saved using the `-o` option
