# Visual Label Testing
This is a project for Hippo Hack Days Sept. 14-16.  The goal is to visually verify correctness of a label image with CV.  

## Installation Notes
If not on Windows, zbar lib needs to be installed manually
```shell
brew install zbar
```
then link it where python can find it
```shell
ln -s $(brew --prefix zbar)/lib/libzbar.dylib /usr/local/lib/
```

See description of approach and examples [here](doc/approach.md)