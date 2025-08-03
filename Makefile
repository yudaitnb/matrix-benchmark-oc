# コンパイラとフラグ設定
CC = gcc
CFLAGS = -O3 -fopenmp -march=native -ffast-math -funroll-loops -fno-trapping-math -falign-loops=32 -fopenmp-simd -ftree-vectorize -mavx2 -mfma
LDFLAGS = -shared -fPIC

# 出力ディレクトリ
BUILDDIR = build
RESULTSDIR = result

# ソースファイルと出力ファイルの対応
SRCS = \
    matmul_naive.c \
    matmul_blocked.c \
    matmul_blocked_omp.c \
    matmul_blocked_omp_tuning.c \

OBJS = $(SRCS:.c=.so)
OBJ_PATHS = $(addprefix $(BUILDDIR)/, $(OBJS))

.PHONY: all clean

# デフォルトターゲット
all: $(BUILDDIR) $(OBJ_PATHS)

# 出力ディレクトリがなければ作成
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ソースファイルをビルドして.soファイルに変換（出力先をBUILDDIRに）
$(BUILDDIR)/%.so: %.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# クリーンターゲット
clean:
	rm -f $(BUILDDIR)/*.so
