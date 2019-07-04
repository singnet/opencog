# Mine Ugly Male Soda Drinker Pattern

Mine 3-conjunct (a.k.a. 3-gram) patterns by incrementally expanding
pattern conjunctions.

## Overview

Given a KB of people and their characteristics, such as `man`,
`woman`, `ugly`, etc.

```scheme
(Inheritance
  (Concept "Allen")
  (Concept "man"))
...
(Inheritance
  (Concept "Davion")
  (Concept "ugly"))
...
(Inheritance
  (Concept "Jenica")
  (Concept "soda drinker"))
```

Mine patterns such as

```scheme
(Lambda
  (Variable "$X")
  (Present
    (Inheritance
      (Variable "$X")
      (Concept "man"))
    (Inheritance
      (Variable "$X")
      (Concept "ugly"))
    (Inheritance
      (Variable "$X")
      (Concept "soda drinker"))))
```

by starting from the top pattern

```scheme
(Lambda
  (Variable "$X")
  (Present (Variable "$X")))
```

finding 1-conjunct (a.k.a. 1-gram) patterns first, such as

```scheme
(Lambda
  (Variable "$X")
  (Present
    (Inheritance
      (Variable "$X")
      (Concept "soda drinker"))))
(Lambda
  (Variable "$X")
  (Present
    (Inheritance
      (Variable "$X")
      (Concept "ugly"))))
```

and combining them to create larger conjunctive patterns such as

```scheme
(Lambda
  (Variable "$X")
  (Present
    (Inheritance
      (Variable "$X")
      (Concept "soda drinker"))
    (Inheritance
      (Variable "$X")
      (Concept "ugly"))))
```

## Usage

Load `ugly-male-soda-drinker.scm` directly into guile to run the whole
thing

```bash
guile -l ugly-male-soda-drinker.scm
```

or paste each scheme commend into guile.

The results should have been stored in the `results` variable, which
can be shown as follows:

```scheme
results
```
