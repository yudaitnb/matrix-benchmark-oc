method MatrixMul(A: seq<seq<int>>, B: seq<seq<int>>) returns (C: seq<seq<int>>)
  requires |A| > 0 && |B| > 0
  requires (forall row :: row in A ==> |row| == |A[0]|)
  requires (forall row :: row in B ==> |row| == |B[0]|)
  requires |A[0]| == |B|
  ensures |C| == |A|
  ensures forall i, j :: 0 <= i < |A| && 0 <= j < |B[0]| ==>
           C[i][j] == sum k | 0 <= k < |A[0]| :: A[i][k] * B[k][j]
{
  var l := |A|;
  var m := |A[0]|;
  var n := |B[0]|;
  C := [];

  // 行方向ループ
  var i := 0;
  while i < l
    invariant 0 <= i <= l
    invariant |C| == i
    decreases l - i
  {
    var row := [];
    var j := 0;
    while j < n
      invariant 0 <= j <= n
      invariant |row| == j
      decreases n - j
    {
      var s := 0;
      var k := 0;
      while k < m
        invariant 0 <= k <= m
        decreases m - k
      {
        s := s + A[i][k] * B[k][j];
        k := k + 1;
      }
      row := row + [s];
      j := j + 1;
    }
    C := C + [row];
    i := i + 1;
  }
}
