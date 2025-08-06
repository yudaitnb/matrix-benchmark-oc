// i行j列の値を、内積の形で再帰的に計算する関数
function Sum(i: int, j: int, A: seq<seq<int>>, B: seq<seq<int>>, k: int): int
  requires |A| > 0 && |B| > 0
  requires 0 <= i < |A| && 0 <= j < |B[0]|
  requires forall row :: row in A ==> |row| == |A[0]|
  requires forall row :: row in B ==> |row| == |B[0]|
  requires |A[0]| == |B|
  requires 0 <= k <= |A[0]|
  decreases k
{
  if k == 0 then 0
  else Sum(i, j, A, B, k - 1) + A[i][k - 1] * B[k - 1][j]
}

// 行列積 C = A * B を計算するメソッド
method MatrixMul(A: seq<seq<int>>, B: seq<seq<int>>) returns (C: seq<seq<int>>)
  // 前提条件：A, B は空でなく、掛け算できるサイズ
  requires |A| > 0 && |B| > 0
  requires forall row :: row in A ==> |row| == |A[0]|
  requires forall row :: row in B ==> |row| == |B[0]|
  requires |A[0]| == |B|

  // 保証条件：C のサイズと中身が正しい行列積になっている
  ensures |C| == |A|
  ensures forall i :: 0 <= i < |A| ==> |C[i]| == |B[0]|
  ensures forall i, j :: 0 <= i < |A| && 0 <= j < |B[0]| ==> C[i][j] == Sum(i, j, A, B, |A[0]|)
{
  var l := |A|;       // 行数（Aの行）
  var m := |A[0]|;    // Aの列数 = Bの行数
  var n := |B[0]|;    // 列数（Bの列）
  C := [];

  var i := 0;
  while i < l
    invariant 0 <= i <= l
    invariant |C| == i
    invariant forall ii :: 0 <= ii < i ==> |C[ii]| == n
    invariant forall ii, jj :: 0 <= ii < i && 0 <= jj < n ==> C[ii][jj] == Sum(ii, jj, A, B, m)
    decreases l - i
  {
    var row := [];

    var j := 0;
    while j < n
      invariant 0 <= j <= n
      invariant |row| == j
      invariant forall jj :: 0 <= jj < j ==> row[jj] == Sum(i, jj, A, B, m)
      decreases n - j
    {
      var s := 0;

      var k := 0;
      while k < m
        invariant 0 <= k <= m
        invariant s == Sum(i, j, A, B, k)
        decreases m - k
      {
        // Aのi行k列と、Bのk行j列を掛けて足す
        s := s + A[i][k] * B[k][j];
        k := k + 1;
      }

      // j列目の計算が終わったら、行に追加
      row := row + [s];
      j := j + 1;
    }

    // i行目の計算が終わったら、Cに追加
    C := C + [row];
    i := i + 1;
  }
}
