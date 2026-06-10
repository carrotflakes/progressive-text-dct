# DCTベースのプログレッシブ・テキスト圧縮 — 実験レポート

(実験進行中 — 各Phase完了ごとに更新)

## 1. 実験設定の要約

- **ベースLM**: Qwen/Qwen2.5-0.5B（hidden=896, 24層, 語彙152k, tied embeddings）。ロード成功のためフォールバック(gpt2)は不使用
- **データ**: Salesforce/wikitext (wikitext-103-raw-v1)。32〜128トークンのチャンクに分割。train 200,000 / validation 2,000 / test 2,000（val/testはそれぞれvalidation/test splitから取得しリークを回避）。平均チャンク長 ~80トークン
- **圧縮**: 系列方向の直交DCT(type-2, norm='ortho')、先頭K係数を保持。K_max=64
- **デコーダ**: LoRA(r=16, α=32, 全attention+MLP線形層) + 入力プロジェクション/係数インデックス埋め込み/長さ埋め込み/BOSをフル学習。学習対象 8.8M / 全503M (1.75%)
- **ハードウェア**: 開発・サニティはローカル RTX 4070 (16GB)。**本学習はRunPod RTX 4090 (24GB) を予定**（コストと速度のバランスが良く、0.5Bモデルには24GBで十分なため。後述）
- **精度**: 凍結ベース重みbf16、forward/backwardはbf16 autocast、LoRA/extrasパラメータはfp32マスター
- **学習時間**: TODO（本学習後に記入）

### Phase 1 サニティチェック結果 ✅

100サンプルをK=64固定で過学習させ、**復元トークン精度 token_acc=1.0000、完全一致 100/100、train loss 4.29→0.0001**。スケール正規化(h_scale)・長さ条件埋め込み・マスク処理が正しいことを確認（`results/phase1_sanity.json`）。これによりPhase 2へ進める状態。

### 単体テスト ✅

DCT実装をscipy.fft.dct(type-2, norm='ortho')と比較し1e-10精度で一致、直交往復・切り詰め誤差の単調性を確認（`src/test_dct.py`）。

## 2. 品質 vs K 曲線

- TODO: results/curves.png

## 3. 定性サンプル

- TODO: results/samples.md 参照

## 4. 仮説の判定

- H1 (プログレッシブ性): TODO
- H2 (グレースフルな劣化): TODO
- H3 (学習デコーダの必要性): TODO
- H4 (DCT順序の意味): TODO

## 5. encoder_layer=0 と 4 の比較

- TODO

## 6. 設計判断ログ

実装中に独断で行った設計判断と理由:

1. **隠れ状態のグローバルスケール正規化 (h_scale)**: エンコーダ隠れ状態のRMSをキャリブレーションバッチ(256件)で測定し、単一スカラーで除算してからDCTをかける。DCT係数(特に第0係数は平均×√nでスケールが大きい)をプロジェクション層に入れる際の数値レンジを抑え、学習初期の不安定を避けるため。スカラー1個なので情報は失われない。
2. **val/testの取得元**: タスクは train split からの分割とも読めるが、リークを避けるため validation 2000件は validation split、test 2000件は test split から取得した。
3. **チャンク長**: 32〜128トークンの一様乱数でストリームを切断。長さ条件埋め込みの全レンジを学習で踏ませるため。
4. **プレフィックスへの注意マスク**: 係数トークン列にも通常の因果マスクを適用(z_kはz_{<k}のみ参照)。双方向化はLM改造が必要なため見送り。
5. **BOS**: Qwen2.5はBOSトークンを持たないため、学習可能なBOSベクトル(d_model次元)を導入。
6. **係数インデックス埋め込みのテーブルサイズ**: B3はランダムなK個の係数(インデックス最大n-1=127)を使うため、テーブルはK_max=64ではなくn_max=128エントリで全バリアント共通とした。
7. **生成終了条件**: 圧縮表現は(Z, n)の組であり nは既知なので、EOSではなく「ちょうどnトークン生成」で打ち切る。
8. **B1の最近傍復号**: コサイン類似度で語彙埋め込みとマッチング。enc0系のB1は生埋め込み(正規化なし)を使用。
9. **エンコーダの部分フォワード**: 第L層までで早期打ち切りする自前実装。起動時にフルフォワードとの一致を自動検証し、不一致なら自動でフルフォワードにフォールバックする。
10. **学習精度**: 重みはfp32で保持し、forward/backwardはbf16 autocast。LoRA学習の数値安定性のため。

## 7. 失敗・予想外の挙動

- datasets 5.0 では `wikitext` が名前空間必須になっており `Salesforce/wikitext` に変更が必要だった。
- **ハードウェア制約によるスケール調整**: 当初のローカルGPUは RTX 4070 (16GB)。task.md既定の20,000step×eff.batch32は実測1.76s/step換算で**1本約19時間・4本約78時間**となり「数時間以内」を大幅超過。コスト最適化のため**本学習はRunPod RTX 4090 (24GB)で実施**する方針(A6000より安価かつbf16演算が高速、0.5Bモデルには24GBで十分)。VRAMは語彙152kのロジットが支配的でmicro_batch=16でも~16GB使うため、`config.yaml`を24GB向け(micro_batch=16, grad_accum=2でeff.batch32, steps=3000≈0.5エポック)に設定し予算重視に縮小した。step数はCLI/スクリプトから可変(`run_all.sh STEPS`)。縮小の理由と手順はREADMEに明記。
- enc0(encoder_layer=0=生埋め込み)のh_scaleは0.015と非常に小さい(生埋め込みのRMS)。グローバルスカラー正規化により学習レンジは他バリアントと揃う。
- evaluate.py検証で、**学習なしB1ベースライン(逆DCT+最近傍)がK=64でacc=0.99に達する**ことを確認。これはn≤Kのチャンクで全係数が保持され生埋め込みが完全復元されるため。小K(K≤8)ではacc<0.18に破綻し、H3の予想挙動と整合。
- TODO: 本学習中に発見したことを追記

## 8. 再現方法

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch transformers peft datasets scipy sentence-transformers matplotlib sacrebleu pyyaml accelerate
scripts/run_sanity.sh   # Phase 1 サニティチェック
scripts/run_train.sh    # Phase 2 学習 (main/enc0/b2/b3)
scripts/run_eval.sh     # Phase 3 評価
```
