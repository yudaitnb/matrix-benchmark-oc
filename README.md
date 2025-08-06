# プログラミング言語を科学する —表現する、処理する、証明する— 
[発表資料など](https://prg.is.titech.ac.jp/ja/news/tanabe-presents-research-introduction-at-open-campus-2025/)

本リポジトリでは発表中に使用した全てのプログラムを保管しています。

# 表現する
Microsoft Excelで適当なデータを用意し、埋めたい箇所でCtrl+Eを押してみてください。
- [Excel でフラッシュ フィルを使用する](https://support.microsoft.com/ja-jp/office/excel-%E3%81%A7%E3%83%95%E3%83%A9%E3%83%83%E3%82%B7%E3%83%A5-%E3%83%95%E3%82%A3%E3%83%AB%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%99%E3%82%8B-3f9bcf1e-db93-4890-94a0-1578341f73f7)

## 論文など
- [Automating String Processing in Spreadsheets using Input-Output Examples (POPL'11)](https://www.microsoft.com/en-us/research/publication/automating-string-processing-spreadsheets-using-input-output-examples/)
  - FlashFillの初出
- [From program verification to program synthesis (POPL'10)](https://dl.acm.org/doi/10.1145/1706299.1706337)
  - Strassenの行列積の実装を自動で合成することに成功

# 処理する
## 必要なパッケージ
- Python処理系 - 下のuvから簡単にインストールできます。
- [NumPy](https://numpy.org/ja/install/)
- [gcc](https://gcc.gnu.org/)
- [uv](https://docs.astral.sh/uv/guides/install-python/)

## Cプログラムのビルド
本リポジトリのトップレベルで以下のコマンドを実行してください。
```shell
$ make
```

## Python環境の準備
```shell
$ uv venv
$ source .venv/bin/activate  # Windowsなら: .venv\Scripts\activate
$ uv pip install numpy matplotlib
```

## ベンチマーク
```shell
$ python3 benchmark.py
```
`matmul_blocked_omp_tuning.c`はRyzen 7950Xに最適化しているので、エラーが出る可能性が高いです。

その後、以下でグラフを出力します。
```shell
$ python3 plot.py
```

# 証明する
## CrossHair
[Yusuke Miyazaki](https://www.ymyzk.com/)さんによって作成された[CrossHair Playground](https://crosshair-web.org/?crosshair=0.1&python=3.8)から試すのが最も楽です。
`verification_crosshair.py`の内容を貼り付けてRunを押すとシンボリック実行による反例探索が行われます。

## Dafny
基本的にはVSCode拡張のDafny拡張を入れれば処理系がついてきます。しかし、Windows環境でないと付属パッケージのインストールがやや面倒だと思います。
- [Dafny - Installation](https://dafny.org/latest/Installation)

拡張が入ったら、VSCode上で`verification_MatrixMul.dfy`を開いてください。自動で検証が走ります。

## 論文など
- [Dafny: an automatic program verifier for functional correctness](https://dl.acm.org/doi/10.5555/1939141.1939161?utm_source=chatgpt.com)
  - Dafnyの論文
- [Lecture slides](https://dafny.org/teaching-material/)
  - Dafnyのチュートリアルなど
- [Dafny Workshop](https://popl25.sigplan.org/series/dafny)
  - Dafnyのワークショップ
