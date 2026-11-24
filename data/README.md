# Data

The spectra are one-dimensional, in the observed frame, on a common grid of 4,096 points between
3,850 and 9,000 Angstrom, stored as `torch.load`-able `.pt` files. The binary files are not part of
the repository.

```
data/
  object-split/     train.pt  val.pt  test.pt     training, validation and test splits
  rare_unknown.pt                                 rare classes, never used for training the classifier
```

Each split is a dictionary `{X: FloatTensor [N, 1, L], y: list[str], meta: list[dict]}`; `meta`
carries `obj_id, file_path, class, phase, phase_known, z, z_known, instrument`. The splits are by
object: every spectrum of an object is in the same split. The test split holds 3,394 spectra of
1,666 objects. The rare set holds 60 objects in nine classes (FBOT, ILRT, luminous red nova,
calcium-rich transient, afterglow, long GRB, QSO, blazar, BL Lac) and 47 novae, which the novelty
analysis treats as ordinary objects.

The spectra come from the ZTF collaboration through SkyPortal and are prepared with
[IRIS](https://github.com/applecider-ml/IRIS). They are available on request; place the files in
this directory following the layout above.
