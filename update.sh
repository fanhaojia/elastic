#!/usr/bin/bash

git init 
git remote remove origin
git remote add origin git@github.com:fanhaojia/elastic.git
git add *
git commit -m "first"
git push origin main 


