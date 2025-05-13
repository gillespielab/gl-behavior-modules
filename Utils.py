# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:30:57 2025

@author: Violet
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import deque
import platform

"""Constants"""
slash = '\\' if platform.system() == 'Windows' else '/' #  / works for both linux and mac


# Read a File
def readfile(filepath:str) -> str:
    """Read a File as a Single String"""
    filestring = None
    with open(filepath, 'r') as file:
        filestring = file.read()
        file.close()
    return filestring

# Read a File by Lines
def readlines(filepath:str, remove_trailing_newlines:bool = True) -> list:
    """Read a File Line by Line"""
    lines = None
    with open(filepath, 'r') as file:
        lines = [line[:-1] if remove_trailing_newlines and line.endswith('\n') else line for line in file.readlines()]
        file.close()
    return lines

# Write a File
def writelines(filepath:str, lines:list, line_separator:str = '\n'):
    """Write a File Line by Line (line separators are added)"""
    with open(filepath, 'w') as file:
        file.write(line_separator.join(lines))
        file.close()

# Find Every File in a Given Directory
def FindFiles(path:str = None, recursive:bool = True):
    """list all files in the specified directory"""
    path = os.getcwd() if path is None else path
    folders = []
    try:
        for filename in os.listdir(path):
            filepath = path + slash + filename
            if os.path.isfile(filepath):
                yield filepath
            elif recursive and os.path.isdir(filepath):
                folders.append(filepath)
    except:
        print(f"error finding files in '{path}'")
    
    for folder in folders:
        yield from FindFiles(folder, recursive)

def Search(target:str, path:str = None, name_only:bool = True, recursive:bool = True):
    """search for file(s) matching a target string (returns a generator)"""
    if name_only:
        return (f for f in FindFiles(path, recursive) if target in f.split(slash)[-1])
    else:
        return (f for f in FindFiles(path, recursive) if target in f)

def search(target:str, path:str = None, name_only:bool = True, recursive:bool = True) -> list:
    """search for file(s) matching a target string"""
    return list(Search(target, path, name_only, recursive))

def FindInFiles(target:str, path:str = None, check_chars:int = 20, recursive:bool = True, verbose:bool = 1, ignore_suffixes = {'dll', 'exe', 'a', 'tar', 'tgz', 'gz', 'zip', 'jpg', 'png', 'mp3', 'mp4', 'mov', 'avi', 'wav', 'dpp', 'eli', 'lib', 'gif', 'svg', 'jar'}):
    """find a target string in Files (Generator)"""
    for filepath in FindFiles(path, recursive):
        if filepath.split('.')[-1].lower() not in ignore_suffixes:
            hit = None
            try:
                with open(filepath, 'r') as file:
                    start = file.read(check_chars)
                    if all(ord(c) < 256 for c in start):
                        if target in start or target in file.read():
                            hit = filepath
                    file.close()
            except PermissionError:
                if verbose & 1: print(f"Permission Denied: '{filepath}'")
            except:
                if verbose & 2: print(f"error reading: '{filepath}'")
            if hit:
                yield hit

def timestamp(mode:int = 3):
    def n_digit(x:int, n:int = 2):
        return str(x + 10**n)[-n:]
    
    now = time.localtime()
    D = f"{now.tm_year}{n_digit(now.tm_mon)}{n_digit(now.tm_mday)}"
    T = f"{n_digit(now.tm_hour)}{n_digit(now.tm_min)}{n_digit(now.tm_sec)}"
    
    res = D if mode&2 else ''
    res += '_'*(mode == 3)
    res += T if mode&1 else ''
    
    return res


def Round(x:float, ux:float, string:bool = True):
    """round a number and its uncertainty to the appropriate number of sig figs (returning the result as a string if requested)"""
    # compute the power of 10 to round to
    p = -int(np.floor(np.log10(ux)))
    
    # round both x and its uncertainty
    if p <= 0:
        x, ux = int(round(x, p)), int(round(ux, p))
    else:
        x, ux = round(x, p), round(ux, p)
    
    # check if rounding changed the power of 10 to round to (re-rounding if necessary)
    if p != -int(np.floor(np.log10(ux))):
        return Round(x, ux, string)
    
    # return the result in the requested format
    if string:
        
        if string in (1, True):
            return f"{x} ± {ux}"
        elif string == 2:
            ud = int(round(ux * 10**p)) if ux < 1 else ux
            return f"{x}({ud})"
        else:
            raise ValueError(f"Unrecognized Format Code '{string}'")
    else:
        return x, ux

def weighted_average(X:list, UX:list = None) -> tuple:
    """compute the weighted average of measurements (X) given their uncertainties (UX)"""
    if UX == None:
        UX = [x[1] for x in X]
        X = [x[0] for x in X]
    x = sum(x / ux**2 for x, ux in zip(X, UX)) / sum(1 / ux**2 for ux in UX)
    ux = np.sqrt(sum(ux**2 for ux in UX)) / len(UX)
    return x, ux

def X2(residuals:list, params:list, digits:int = 2):
    """compute the Reduced χ² from the Studentized Residuals and the List of Fit Parameters"""
    ddof = len(residuals) - len(params)
    return round(np.sum(residuals**2) / ddof, digits) if ddof > 0 else 'N/A'

def smooth(x:list, width:int, symmetric:bool = True, kernel = None):
    """Use a Sliding Window to Smooth a Dataset (kernel = None defaults to the average of the window)
    
    If symmetric == True the actual width will be 2*width + 1 
    (i.e. <width> is interpreted as the half-width)
    """
    w = 2 * width + 1 if symmetric else width
    window = deque(x[:w - 1])
    window_sum = sum(window)
    smoothed = []
    for i in range(width, len(x)):
        window.append(x[i])
        window_sum += x[i]
        smoothed.append(kernel(window) if kernel else window_sum / w)
        window_sum -= window.popleft()
    return smoothed

def binary_mean(k:int, n:int = None) -> tuple:
    """k = sum(success), n = total_trials -> mean, uncertainty on the mean
    
    alternatively, you can pass k as the raw data (e.g. k = [0,1,1,0,0,1,0,1,...]) 
    and leave n as None, and it will automatically calculate k and n
    
    mean = k / n
    uncertainty on the mean = sqrt(k * (n - k) / n^3) = sqrt(var / n)
    
    Can be Vectorized (if k and n are vectors of the same shape, or if 
    either is a scalar)
    """
    
    # Compute the Relavent Stats
    if n == None:
        n = len(k)
        k = sum(k)
    
    # Compute and Return the Mean/Uncertainty
    return k / n, np.sqrt((k * (n - k)) / n**3)

# Build a Set of Common Charaters
class Chars:
    def __init__(self, blank:bool = False):
        """Initializes a Set of Common Characters (or a completely blank object if blank==True)"""
        if not blank:
            for name, val in {'pm':'±', 'squared':'²', 'angstrom':'Å', 'hbar':'ℏ', 'l':'ℓ', 'emf':'ℰ', 'prop':'∝', 'approx':'≈', 'ne':'≠', 'scale':'~', 'del':'∇', 'inf':'∞', 'hat':'\u0302', 'flat':'♭', 'natural':'♮', 'sharp':'♯'}.items():
                self.__dict__[name] = val
            for c, name in enumerate(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'bad', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']):
                self.__dict__[name] = chr(c + 945)
                self.__dict__[name.capitalize()] = self.__dict__[name].upper()
            
            self.vectors = Chars(True)
            for name in 'ijkrnt':
                self.vectors.__dict__[name] = name + self.hat

chars = Chars()
