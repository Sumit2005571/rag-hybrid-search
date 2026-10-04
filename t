[33m7c90045[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m)[m ﻿feat: implement multi-format document ingestion and modular chunking pipeline
 .env.example                |   5 [32m+[m
 README.md                   | 229 [32m++++++++++++[m[31m-[m
 app/chunking/__init__.py    |   3 [32m+[m[31m-[m
 app/chunking/base.py        | 100 [32m+++++[m[31m-[m
 app/chunking/fixed.py       |  95 [32m+++++[m[31m-[m
 app/chunking/recursive.py   | 255 [32m++++++++++++++[m[31m-[m
 app/chunking/semantic.py    | 327 [32m+++++++++++++++++[m[31m--[m
 app/config.py               |  17 [32m+[m
 app/ingestion/__init__.py   |  24 [32m+[m[31m-[m
 app/ingestion/loaders.py    | 317 [32m++++++++++++++++[m[31m--[m
 app/ingestion/metadata.py   |  44 [32m++[m[31m-[m
 app/ingestion/normalizer.py |  44 [32m++[m[31m-[m
 requirements.txt            | Bin [31m2468[m -> [32m1187[m bytes
 scripts/chunk.py            | 266 [32m+++++++++++++++[m
 scripts/ingest.py           | 140 [32m++++++++[m
 tests/fixtures/sample.html  |  24 [32m++[m
 tests/fixtures/sample.md    |  21 [32m++[m
 tests/fixtures/sample.pdf   | Bin [31m0[m -> [32m1394[m bytes
 tests/fixtures/sample.txt   |  10 [32m+[m
 tests/test_chunking.py      | 763 [32m++++++++++++++++++++++++++++++++++++++++++++[m
 tests/test_ingestion.py     | 409 [32m++++++++++++++++++++++++[m
 tests/test_structure.py     |   2 [32m+[m[31m-[m
 22 files changed, 2997 insertions(+), 98 deletions(-)
