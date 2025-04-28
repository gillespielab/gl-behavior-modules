# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:25:33 2025

@author: Violet
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
try:
    from Modules.Utils import search, readlines
except:
    from Utils import search, readlines
try:
    from Modules.ParameterFileInterface import ParameterFile, Parser
except:
    from ParameterFileInterface import ParameterFile, Parser
import platform
import time
import re
from multiprocessing import Pool

"""
Constants
"""
slash = '\\' if platform.system() == 'Windows' else '/' #  works for windows, linux, and mac
home_well = 0   #  the index of the home well (this is true regardless of port assignment)


"""
Utils
"""

# Get a DateTime Stamp
def get_timestamp(format_str = "%Y%m%d_%H%M%S") -> str:
    """Get a Timestamp Formatted Using the Given Format String
    
    See doccumentation for the time.py library for how to 
    construct a format string:
        https://docs.python.org/3/library/time.html for 
    """
    return time.strftime(format_str.replace("%f", str(int(round(time.time()%1 * 10**6)))))

# Class for Making Discrete Color Gradients
class Colors:
    rainbow = [
        np.array([236,  97, 115]),
        np.array([241, 174,  74]),
        np.array([225, 221, 110]),
        np.array([ 81, 163,  80]),
        np.array([105, 100, 255]),
        np.array([189,  99, 190])
    ]
    
    def to_hex(color:np.ndarray) -> str:
        return '#' + ''.join(hex(int(round(c)))[2:] for c in color)
    
    def gradient(n:int, colors:list = rainbow, use_hex:bool = True) -> list:
        """Make an n-Color Linear Gradient Using the Color List Provided"""
        def linear_interp(x, x1, x2, y1, y2):
            f = (x - x1) / (x2 - x1)
            return f * y1 + (1 - f) * y2
        
        X = np.linspace(0, len(colors) - 1, n)
        i = 0
        grad = []
        for x in X:
            while x > i + 1:
                i += 1
            grad.append(linear_interp(x, i, i + 1, colors[i + 1], colors[i]))
        
        if use_hex:
            grad = [Colors.to_hex(color) for color in grad]
        
        return grad


# An Object for Storing Raw Log File Lines
class Line:
    # Main Parser
    parser = Parser("[{int}:{int}:{float}] {str}")
    
    # Common Line Parsers (to use after main parsing)
    command_parser = Parser("SCQTMESSAGE: {str};")  #  commands to StateScript
    state_parser = Parser("{int} {int} {int}")      #  state updates from StateScript
    callback_parser = Parser("{int} {str} {int}")   #  callbacks from StateScript (regex overridden)
    callback_parser.regex = re.compile("^\d+ (UP|DOWN|LOCKOUT|LOCKEND)")
    poke_parser = Parser("well {int} poked; reward given = {int}")
    
    def __init__(self, line:str):
        # Parse the Line
        self.line_raw = line
        self.hour, self.minute, self.second, self.text = Line.parser(self.line_raw)
        self.timestamp = (self.hour, self.minute, self.second)
        
        # Initialize Common Line Type Attributes
        self.command = None
        self.state = None
        self.callback = None
        self.poke = None
        
        # Check for Common Line Types
        if Line.command_parser.match(self.text):
            self.command = Line.command_parser(self.text)
        elif Line.state_parser.match(self.text):
            self.state = Line.state_parser(self.text)
        elif Line.callback_parser.match(self.text):
            self.callback = Line.callback_parser(self.text)
        elif Line.poke_parser.match(self.text):
            self.poke = Line.poke_parser(self.text)
        
        # Common Line Type Flags
        self.is_command = self.command != None
        self.is_state_update = self.state != None
        self.is_callback = self.callback != None
        self.is_down = self.is_callback and self.callback[1] == 'DOWN'
        self.is_poke = self.poke != None
    
    def __repr__(self):
        return self.line_raw

class Stats:
    def __init__(self, goal = 0, other = 0, lock = 0, home = 0):
        self.goal = goal
        self.other = other
        self.lock = lock
        self.home = home
        self.computed = False
    
    def goal_rate(self, default = -1.0):
        d = self.goal + self.other
        return self.goal / d if d else default
    
    def __repr__(self):
        return f"home = {self.home}\ngoal = {self.goal}\nother = {self.other}\nlock = {self.lock}\ngoal rate = {round(self.goal_rate(), 3)}"

"""
Objects for Containing the Parsed Data in a Convenient Format, and for Calulating/Plotting Results
"""

class Poke:
    parser = Parser("poke: ({int}, {bool}, {bool}, {int}, [{[int, ', ']}], {int})")
    
    current = None
    
    def __init__(self, well:int, rewarded:bool, search_mode:bool, maze_phase:int, goal:list, start_time:int = None, end_time:int = None, trial = None):
        """A Data Object for a Single Reward Well Poke"""
        
        Poke.current = self
        
        self.well = well
        self.goal = goal
        self.rewarded = rewarded
        self.phase = maze_phase
        self.search_mode = search_mode
        self.is_home = well == home_well
        self.trial = Trial.current if trial == None else trial
        self.start = start_time
        self.end = end_time
    
    def __repr__(self):
        return f"Poke[Well = {self.well}, Rewarded = {1 if self.rewarded else 0}]"
    
    def __str__(self):
        return f"Poke[{self.well} {1 if self.rewarded else 0}]"
    
    def __hash__(self):
        return id(self)
    
    def get_table_data(self) -> tuple:
        """Return Data to Store in the DB Table"""
        return self.well, self.rewarded, self.start, self.end
    
    def raster_plot(self, x:int, axes:plt.axes, lockout:bool = False) -> int:
        """
        Add the Poke to a Raster Plot of the Raw Data

        Parameters
        ----------
        x : int
            The x position to draw this poke at.
        axes : plt.axes
            The matplotlib axes to draw the poke on.

        Returns
        -------
        int
            The x position of the next poke.

        """
        
        # Get the Formatting Params
        color = 'grey' if lockout else ('blue' if self.search_mode else 'red')
        fmt = '.' if self.rewarded else '+'
        
        # Add the Poke
        axes.errorbar(x, self.well, fmt = fmt, color = color)
        
        # Return the Next Position
        return x + 1

class Trial:
    
    current = None
    trial_num = 1
    
    def __init__(self, block = None, home_poke:Poke = None, outer_poke:Poke = None, lockout_pokes:list = None):
        
        if Trial.current: Trial.current._on_load()
        Trial.current = self
        
        self.home = home_poke
        self.outer = outer_poke
        self.lockouts = [] if lockout_pokes == None else lockout_pokes
        self.block = block
        self.index = len(block.trials) if block else -1
        self.trial_num = Trial.trial_num
        Trial.trial_num += 1
        self.reps_remaining = -1
        self.search_mode = 0
        
        self.goal = self.block.goal if self.block else None
        
        self.start = home_poke.start if home_poke else None
        self.end = None
        self.complete = False
        self.rewarded = False
        
        self.lines = []
        
        # Update the Trial Pointers
        if self.home: self.home.trial = self
        if self.outer: self.outer.trial = self
        if self.lockouts:
            for poke in self.lockouts: poke.trial = self
    
    def __repr__(self) -> str:
        if self.complete:
            lockouts = [poke.well for poke in self.lockouts]
            if lockouts:
                return f"Trial[{self.outer.well} {int(self.rewarded)} lockouts={lockouts}]"
            else:
                return f"Trial[{self.outer.well} {int(self.rewarded)}]"
        else:
            return "Trial[Incomplete]"
    
    def has_pokes(self) -> bool:
        """A Basic Check for if the Trial Contains Any Data"""
        return self.home or self.outer or self.lockouts
    
    def add_home(self, poke:Poke = None) -> None:
        """Add a Home Poke"""
        self.home = poke if poke else Poke.current
        self.search_mode = self.home.search_mode
        self.start = self.home.start
        
        # Update the Goal
        if not self.goal and self.block: self.goal = self.block.goal
    
    def add_outer(self, poke:Poke = None) -> None:
        """Add an Outer Poke"""
        # Add the Poke and Update the Flags
        self.outer = poke if poke else Poke.current
        self.complete = bool(self.home)
        self.rewarded = self.outer.rewarded
        
        # Update the Goal
        if not self.goal and self.block: self.goal = self.block.goal
        
        # Switch the Associated Goal Block
        if not self.block.first_trial_adjusted and self.index == 0 and Block.previous and self.outer.well in Block.previous.goal:
            self.block.first_trial_adjusted = True
            self.index = len(Block.previous.trials)
            self.search_mode = 0
            self.home.search_mode = 0
            self.outer.search_mode = 0
            self.reps_remaining = 0
            self.block.trials.pop(0)
            self.block = Block.previous
            self.block.trials.append(self)
            for i, trial in enumerate(Block.current.trials):
                trial.index = i
    
    def add_lockout(self, poke:Poke = None) -> None:
        """Add a Lockout Poke"""
        # Add the Lockout
        self.lockouts.append(poke if poke else Poke.current)
        
        # Update the Goal
        if not self.goal and self.block: self.goal = self.block.goal
    
    def to_table_entry(self, index:int = None, trial_num:int = None, include_index:bool = False) -> list:
        """(index), trial_num, start_time, end_time, outer_reward, search_mode, outer_well, goal_wells, reps_remaining, leave_home, outer_time, leave_outer, lockouts"""
        include_index |= index != None
        return [self.index if index == None else index]*bool(include_index) + [
            self.trial_num if trial_num == None else trial_num, 
            self.start, 
            self.end, 
            int(self.rewarded), 
            int(self.search_mode), 
            self.outer.well if self.outer else None, 
            self.goal, 
            self.reps_remaining, 
            self.home.end if self.home else None, 
            self.outer.start if self.outer else None, 
            self.outer.end if self.outer else None, 
            [(poke.well, poke.start, poke.end) for poke in self.lockouts]
        ]
    
    def from_table_entry(block, entry:list):
        """(index), trial_num, start_time, end_time, outer_reward, search_mode, outer_well, goal_wells, reps_remaining, leave_home, outer_time, leave_outer, lockouts"""
        # Poke(well, rewarded, search_mode, maze_phase, goal, start_time, end_time, trial)
        trial = Trial(block)
        trial.index = entry[0]
        trial.trial_num = entry[1]
        trial.start = entry[2]
        trial.end = entry[3]
        trial.rewarded = entry[4]
        trial.goal = entry[7]
        trial.reps_remaining = entry[8]
        trial.add_home(Poke(home_well, True, entry[5], 0, entry[7], trial.start, entry[9], trial))
        if entry[6] != None:
            trial.add_outer(Poke(entry[6], trial.rewarded, entry[5], 1, entry[7], entry[10], entry[11], trial))
        phase = 0
        for well, t_start, t_end in entry[12]:
            trial.add_lockout(Poke(well, 0, entry[5], phase, entry[7], t_start, t_end, trial))
            phase = 2
        
        trial.complete = trial.outer != None # trial._on_load()
        
        return trial
    
    def _on_load(self):
        """Code to Run After Loading the Data"""
        # Update the Start Time
        if self.home: self.start = self.home.start
        
        # Update the Completion Flag
        self.complete = bool(self.home) and bool(self.outer)
        
        # Update the Rewarded Flag
        self.rewarded = bool(self.outer and self.outer.rewarded)
        
        # Update the End Time
        if self.lockouts:
            self.end = self.lockouts[-1].end
        elif self.outer:
            self.end = self.outer.end
        elif self.home:
            self.end = self.home.end
        
        # Update the Goal
        if not self.goal and self.block: self.goal = self.block.goal
        
        # Update the Search Mode
        if self.home:
            self.search_mode = self.home.search_mode
    
    def _load_from_log(self, lines:list, start:int) -> int:
        self.home = None
        self.outer = None
        self.lockouts.clear()
        
        # Find the Start of the Trial
        i = start
        while i < len(lines) and lines[i].text != 'well 0 poked; reward given = 1':
            i += 1
        self.lines.append(lines[i])
        i += 1
        
        # Parse the Pokes
        poke = None
        while i < len(lines) and lines[i].text != 'well 0 poked; reward given = 1':
            # Record the Line
            self.lines.append(lines[i])
            
            # Look for Special Lines
            if lines[i].is_callback:
                if lines[i].callback[1] == 'DOWN' and poke:
                    poke.end = lines[i].callback[0]
                elif lines[i].callback[1] == 'UP':
                    self.end = lines[i].callback[0]
            elif Poke.parser.match(lines[i].text):
                poke = Poke(*Poke.parser(lines[i].text), None, self)
                
                if not self.home:
                    self.add_home(poke)
                elif not self.outer:
                    self.add_outer(poke)
                else:
                    self.add_lockout(poke)
            
            # Increment the Line Index
            i += 1
        
        # Update the Flags
        self._on_load()
        
        # Return the Index of the Next UP Poke
        return i
    
    def raster_plot(self, x:int, axes:plt.axes, included:int = 0):
        """
        Add the Poke to a Raster Plot of the Raw Data
    
        Parameters
        ----------
        x : int
            The x position to draw this poke at.
        axes : plt.axes
            The matplotlib axes to draw the poke on.
        included : int (bitmask), optional
            Whether or not to include lockouts in the plot (included&1).
            Whether or not to include home pokes in the plot (included&2).
            The default is 0.
    
        Returns
        -------
        int
            The x position of the next poke.
    
        """
        # Plot the Home Poke
        if included&2 and self.home:
            x = self.home.raster_plot(x, axes)
        
        # Plot the Outer Poke
        if self.outer:
            x = self.outer.raster_plot(x, axes)
        
        # Plot Lockout Pokes
        if included&1 and self.lockouts:
            for poke in self.lockouts:
                x = poke.raster_plot(x, axes, True)
        
        # Return the Final Position
        return x

class Phases:
    search = 1
    perseverative_search = 2
    good_repetition = 4
    lapse = 8
    bad_repetition = perseverative_search | lapse
    repeat_with_lapses = good_repetition | lapse
    repeat = perseverative_search | repeat_with_lapses
    all = search | repeat

class Block:
    info_parser = Parser("new goal block: [goal = [{[int ', ']}], outreps = {int}]")
    
    current = None
    previous = None
    
    def __init__(self, epoch = None, block_index:int = None):
        
        if Block.current: 
            Block.current._on_load()
            self.previous_goal = Block.current.goal
        if Block.previous and not Block.previous.complete:
            Block.previous._on_load()
        Block.previous = Block.current
        Block.current = self
        self.previous_goal = None
        
        self.index = block_index if block_index != None else len(epoch.blocks)
        self.epoch = Epoch.current if epoch == None else epoch
        
        self.goal = None
        self.prev_goal = None
        self.outreps = None
        
        self.lines = []
        
        self.all_trials = []
        self.trials = []
        self.phases = None
        self.stats = Stats()
        
        self.start = None
        self.end = None
        
        self.complete = False
        self.first_trial_adjusted = False
        
        self.phase_params = None
    
    def __repr__(self):
        incomplete = '' if self.complete else ': Incomplete'
        return f"Block[{self.epoch.name} {self.epoch.index}-{self.index}{incomplete}]"
    
    def __hash__(self):
        return id(self)
    
    def __len__(self):
        return len(self.trials)
    
    def has_trials(self) -> bool:
        return len(self) > 0
    
    def has_pokes(self) -> bool:
        return any(t.has_pokes() for t in self.trials)
    
    def __bool__(self): 
        return True # if this doesn't exist, then bool(self) = bool(len(self)) = len(self) > 0, which gives weird behavior
    
    def __iter__(self):
        return iter(self.trials)
    
    def _on_load(self, complete:bool = None):
        """Code to Run After Loading the Data"""
        reps_remaining = True
        if complete == None and self.outreps:
            reps_remaining = self.outreps
            for t in self.trials:
                t.reps_remaining = reps_remaining
                if t.rewarded: reps_remaining -= 1
        elif self.trials and self.trials[0].reps_remaining > 0:
            self.outreps = self.trials[0].reps_remaining
            if complete == None:
                self._on_load() # self.outreps evals to True in the recursion since reps_remaining > 0
                return
        
        self.complete = (not reps_remaining) if complete == None else complete
        
        self.compute_stats()
    
    def compute_stats(self) -> tuple:
        """Computes/Returns the Count Stats: (home, goal, other, lock)"""
        self.stats.home = len(self.all_trials)
        self.stats.goal = sum(t.rewarded for t in self)
        self.stats.other = len(self) - self.stats.goal
        self.stats.lock = sum(len(t.lockouts) for t in self)
        return (self.stats.home, self.stats.goal, self.stats.other, self.stats.lock)
    
    def _load_from_log(self, lines:list, start:int) -> int:
        """Read the Data From a File Line-by-Line Starting at the Specified Index, Returning the Next Unused Index"""
        # Parse the Lines
        i = start + 1
        while i < len(lines) and lines[i][0] not in '-=':
            self.lines.append(Line(lines[i]))
            i += 1
        
        # Check if the Block was Completed
        self.complete = i < len(lines) and lines[i][0] == '-'
        
        # Get the Outreps
        self.goal, self.outreps = next((Block.info_parser(line.text) for line in self.lines if Block.info_parser.match(line.text)), None)
        if self.outreps == None:
            print(f'Warning: goal/outreps not found for {str(self)}')
        
        # Get the Trials
        j = 0
        while j < len(self.lines):
            self.all_trials.append(Trial(self))
            j = self.all_trials[-1]._load_from_log(self.lines, j)
            if self.all_trials[-1].complete:
                self.trials.append(self.all_trials[-1])
        
        # Pre-Process the Data
        self._on_load()
        
        # Return the Next Index in the File
        return i + 1
    
    def get_phases(self, included:int, is_good_repetition = None, is_memory_lapse = None) -> list:
        
        # Get the Methods for Determining "Good Repetition" and "Memory Lapse"
        if not hasattr(is_good_repetition, '__call__'):
            param = 3 if is_good_repetition == None else is_good_repetition
            is_good_repetition = lambda trials, i: Block._is_good_repetition(trials, i, param)
        if not hasattr(is_memory_lapse, '__call__'):
            param = 3 if is_memory_lapse == None else is_memory_lapse
            is_memory_lapse = lambda trials, i: Block._is_memory_lapse(trials, i, param)
        
        # Allocate the Trials
        phase = Phases.search
        phases = [(phase, [])]
        for i, trial in enumerate(self.trials):
            if phase == Phases.search:
                # even if the trials is rewarded it's still search phase
                phases[-1][1].append(trial)
                
                # if the trial is rewarded we need to change phase
                if trial.rewarded:
                    phase = Phases.good_repetition if is_good_repetition(self.trials, i) else Phases.perseverative_search
                    phases.append((phase, []))
            elif phase == Phases.good_repetition and not trial.rewarded and is_memory_lapse(self.trials, i):
                # A Memory Lapse Has Started
                phase = Phases.memory_lapse
                phases.append((phase, []))
                phases[-1][1].append(trial)
            elif phase&Phases.bad_repetition and trial.rewarded and is_good_repetition(self.trials, i):
                # The Rat is Repeating Well (Again) - phase&Phases.bad_repetition accounts for both perseverative search and lapses
                phase = Phases.good_repetition
                phases.append((phase, []))
                phases[-1][1].append(trial)
            else:
                # no reason to change phase
                phases[-1][1].append(trial)
        
        # Filter and Return the Result
        return [phase for phase in phases if phase[0] & included]
    
    def _is_good_repetition(trials:list, index:int, mode:int) -> bool:
        if type(mode) == int and 0 < mode < len(trials) - index:
            # n-in-a-row method
            return all(trials[i].rewarded for i in range(index, index + mode))
        elif type(mode) == int and index - len(trials) < mode <= 0:
            # count backwards with n mistakes method
            return sum(trials[i].rewarded for i in range(len(trials) - 1, index - 1, -1)) > len(trials) - index + mode
        else:
            raise ValueError(f"Unrecognized Mode to Determine Good Repetition: {mode}")
    
    def _is_memory_lapse(trials:list, index:int, mode:int) -> bool:
        if type(mode) == int and 0 < mode < len(trials) - index:
            # n-in-a-row method
            return not any(trials[i].rewarded for i in range(index, index + mode))
        else:
            raise ValueError(f"Unrecognized Mode to Determine Memory Lapses: {mode}")
    
    def raster_plot(self, x:int = 1, axes:plt.axes = None, back_color:str = None, alpha:float = 0.2, included:int = 0) -> int:
        """
        Create a Raster PLot of the Goal Block (either on its own if axes == None, or as a piece of a larger plot)

        Parameters
        ----------
        x : int, Optional
            The x position to draw the first poke at.
            The default is 1.
        axes : plt.axes, optional
            The matplotlib axes to draw the poke on. 
            The default is None (create a new plot).
        back_color : str, optional
            The color to draw the background of the goal block.
            The default is None (white)
        alpha : float, optional
            Alters the color of the background of the plot.
            The default is 0.2.
        included : int (bitmask), optional
            Whether or not to include lockouts in the plot (included&1).
            Whether or not to include home pokes in the plot (included&2).
            The default is 0.

        Returns
        -------
        int
            The x position of the first poke of the next block.

        """
        
        # Get the 1st X Position
        x0 = x
        
        # Check if the Plot is a Subplot
        is_not_subplot = axes == None
        
        # Initialize a New Plot
        if is_not_subplot:
            fig, axes = plt.subplots()
        
        # Add the Pokes
        for trial in self.trials:
            x = trial.raster_plot(x, axes, included)
        
        # Add the Background Color
        if back_color:
            axes.add_patch(plt.Rectangle(
                (0 if x0 == 1 else x0 - 0.5, 0.5),
                x - x0,
                self.rat.cohort.study.arms + 1, 
                fill = True, 
                color = back_color, 
                alpha = alpha, 
                zorder = float('inf'), 
                figure = axes.figure
            ))
        
        # Set the Display Parameters and Add Labels
        if is_not_subplot:
            fig.set_figwidth(x / 8)
            fig.set_figheight(2)
            plt.ylabel('Arm Number')
            plt.ylim(0.8 - (included&2), 6.2)
            plt.xlabel('Trial Number')
            plt.xlim(x0 - 1, x)
            plt.show()
        
        # Return the Next X Position
        return x

class _nwb_filename_parser:
    def __init__(self, underscore:bool = False):
        self.underscore = bool(underscore)
        self.regex = r'^.*?\d+' + '_'*self.underscore + '\.nwb'
    
    def parse(self, filename:str, safe:bool = False) -> tuple:
        # Check Safe Mode
        if safe and not self.match(filename):
            # Return that the Filename Doesn't Match
            return None, None
        
        # Get the Rat's Name
        i = 0
        while i < len(filename) and not filename[i].isdigit():
            i += 1
        
        name = filename[:i]
        
        # Get the Date Stamp
        j = i + 1
        while j < len(filename) and filename[j].isdigit():
            j += 1
        
        date = int(filename[i:j])
        
        # Return the Results
        return name, date
    
    def match(self, filename:str):
        return self.regex.match(filename)
    
    def __call__(self, filename:str, safe:bool = False) -> tuple:
        """Parse the Filename (see .parse for Full Documentation)
        
        Uses the regex to match the input string if safe == True
        """
        if safe:
            return self.parse(filename, False) if self.match(filename) else None
        else:
            return self.parse(filename, False)

class Epoch:
    
    rmLockoutParser = Parser("({int},{int},{int})")
    Parser.add_datatype('rmLockout', rmLockoutParser)
    rmParamsParser = Parser("{str} {int} {int} {int} ({tuple[float, ', ']}) {int} {outreps} {int} {int} {int} {int} {int} {int} {threshold} {int} {int}")
    rmTrialParser = Parser("{int}\t{int}\t{int}\t{int}\t{int}\t{int}\t[{[int, ', ']}]\t{int}\t{int}\t{int}\t{int}\t[{[rmLockout, ', ']}]")
    Parser.add_datatype('rmParams', rmParamsParser)
    Parser.add_datatype('rmTrials', rmTrialParser)
    rmTDFileParser = Parser("{rmParams}\n{[rmTrials, '\n']}")
    
    current = None
    
    def __init__(self, filepath:str = None, rat = None, index:int = None):
        
        Epoch.current = self
        
        # Initialize the Main Attributes
        self.name = None
        self.date = None
        self.epoch_number = None
        self.all_trials = []
        self.blocks = []
        self.trials = []
        self.complete = False
        self.lines = None
        self.index = index if index != None else (len(rat.epochs) if rat else 0)
        self.home_well = 0
        self.rat = rat
        self.stats = Stats()
        
        # Initialize File Section Attributes
        self.header = []
        self.parameters = []
        self.initialization = []
        self.params_final = []
        self.footer = []
        self.file_sections = [self.header, self.parameters, self.initialization, self.blocks, self.params_final, self.footer]
        
        # Initialize Path Attributes
        self.filepath = filepath
        self.folder = None
        self.filename = None
        self.folder = None
        self.datetime_loaded = None
        
        # Load the File (if given)
        if filepath != None:
            # Store the Computed Path Attributes
            self.folder = filepath.split(slash)
            self.filename = self.folder[-1]
            self.folder = slash.join(self.folder[:-1])
            
            # Get the File Extensions
            extension = self.filename.split('.')[-1]
            if extension in Epoch._filename_parsers:
                # Try to Match the Filepath
                filename_matched = False
                parser = Epoch._filename_parsers[extension]
                load = Epoch._file_loaders[extension]
                
                # Parse the Filepath
                if parser.match(self.filename):
                    filename_matched = True
                    self.date, self.name, self.epoch_number = parser(self.filename)
                    load(self)
                    self.datetime_loaded = get_timestamp()
                
                if not filename_matched:
                    print(f"Warning: uparsable filename, unable to load '{self.filename}'")
            else:
                print(f"Warning: unrecognized file extension, unable to load '{self.filename}'")
                return
            
            # Get the Pokes
            for block in self.blocks:
                self.all_trials.extend(block.all_trials)
                self.trials.extend(block.trials)
            
            # Compute the Stats
            self.compute_stats()
    
    def __repr__(self) -> str:
        incomplete = '' if self.complete else ' Incomplete'
        return f"Epoch[{self.name} {self.date} {self.epoch_number}{incomplete}]"
    
    def __getitem__(self, key):
        return self.blocks[key]
    
    def __iter__(self):
        return iter(self.blocks)
    
    def __len__(self):
        return len(self.blocks)
    
    def __bool__(self): 
        return True # if this doesn't exist, then bool(self) = bool(len(self)) = len(self) > 0, which gives weird behavior
    
    def has_blocks(self) -> bool:
        return len(self) > 0
    
    def __lt__(self, other):
        # allows you to sort epochs by when they happened (only makes sense when comparing epochs for the same rat)
        return self.get_sortable_key() < other.get_sortable_key()
    
    def get_sortable_key(self):
        """used when sorting epochs by when they occured (only makes sense when comparing epochs for the same rat)"""
        return (self.name, self.date, self.epoch_number)
    
    def compute_stats(self, recompute:bool = False) -> None:
        """Compute the home/goal/other/lock stats for the Epoch (stored in self.stats)"""
        if not self.stats.computed or recompute:
            for trial in self.trials:
                if trial.home: 
                    self.stats.home += 1
                if trial.outer:
                    if trial.outer.rewarded:
                        self.stats.goal += 1
                    else:
                        self.stats.other += 1
                self.stats.lock += len(trial.lockouts)
            self.stats.computed = True
    
    param_vector_attributes = ['goals', 'cues', 'outreps']
    def get_param_vector(epoch) -> tuple:
        """Builds a Parameter Vector from the Attribute Names in Epoch.param_vector_attributes
        
        Default Attributes: ['goals', 'cues', 'outreps']
        
        The attribute names should be attributes of epoch.parameters. Override this method if needed
        """
        return tuple(epoch.parameters.__dict__[attribute] for attribute in Epoch.param_vector_attributes)
    
    def _load_from_rmTableData(self, epoch_index:int = 0):
        """Load an Epoch from a .rmTableData File (as created by the maze)"""
        # Load the Data
        file = None
        with open(self.filepath, 'r') as f:
            file = f.readall()
            f.close()
        
        # Parse the Data
        params, trials = Epoch.rmTDFileParser(file)
        
        # Load the Data
        self.load_from_tables([epoch_index] + params, [[i] + row for i, row in enumerate(trials)])
    
    def _load_from_log(self) -> None:
        """Load Epoch Behavior Data from a Python Log"""
        
        # Read the File
        self.lines = readlines(self.filepath)
        
        # Get the Header
        count = 2
        i = 0
        while count and i < len(self.lines):
            if self.lines[i][0] == '=':
                count -= 1
            else:
                self.header.append(Line(self.lines[i]))
            i += 1
        
        # Get/Parse the Initial Parameter File
        while i < len(self.lines) and (not self.lines[i] or self.lines[i][0] != '='):
            self.parameters.append(self.lines[i])
            i += 1
        parameter_file = ParameterFile()
        parameter_file.load_from_file(self.parameters[1:])
        self.parameters = parameter_file
        i += 1
        
        # Get the Initialization Lines
        while i < len(self.lines) and self.lines[i][0] != '=':
            self.initialization.append(Line(self.lines[i]))
            i += 1
        i += 1
        
        # Parse the Blocks
        while i < len(self.lines) and self.lines[i][0] == '-':
            self.blocks.append(Block(self, None))
            i = self.blocks[-1]._load_from_log(self.lines, i)
        
        # Parse the Final Parameter File
        while i < len(self.lines) and (not self.lines[i] or self.lines[i][0] != '='):
            self.params_final.append(self.lines[i])
            i += 1
        parameter_file = ParameterFile()
        parameter_file = parameter_file.load_from_file(self.params_final[1:])
        self.params_final = parameter_file
        i += 1
        
        # Parse the Footer
        while i < len(self.lines):
            if self.lines[i][0] != '=':
                self.footer.append(Line(self.lines[i]))
            i += 1
    
    def _load_from_ss_log(self) -> None:
        """Load Epoch Behavior Data from a StateScript Log"""
        raise NotImplementedError()
    
    def _load_from_nwb_partial(self, nwb_data:bytes, start:int) -> int:
        """Load Epoch Behavior Data from an Unprocessed nwb File (returns the position of the next unused byte)"""
        raise NotImplementedError()
    
    def _load_from_nwb_partial_(self, nwb_data:bytes, start:int) -> int:
        """Load Epoch Behavior Data from a Processed nwb File (returns the position of the next unused byte)"""
        raise NotImplementedError()
    
    # Filename Parsers -> Data Loaders
    _filename_parsers = {
        'log' : Parser("{int}_{str}_{int}.log"),
        'stateScriptLog' : Parser("{int}_{str}_{int}.stateScriptLog")
    }
    _file_loaders = {
        'log' : _load_from_log,
        'stateScriptLog' : _load_from_ss_log
    }
    _nwb_filename_parsers = [
        (_nwb_filename_parser(False), _load_from_nwb_partial),
        (_nwb_filename_parser(True), _load_from_nwb_partial_)
    ]
    
    # Save the Data to an nwb File
    def save_to_nwb(self, filename:str = None, folder:str = None):
        raise NotImplementedError()
    
    # Make the Trial Table
    def make_TrialInfo(self, include_index:bool = True):
        """Populate the Trial Info Table (currently returns the table as a python list)"""
        return [trial.to_table_entry(i, i + 1, include_index) for i, trial in enumerate(self.all_trials)]
    
    # Make a Parameter Table Entry
    def to_table_entry(self, index:int = None, include_index:bool = True) -> list:
        """(index), name, date, epoch_num, blocks_completed, parameter_file_version, goals, outreps, delay, goal_selection_mode, cues, forageassist, trials, success_threshold, eps_remaining"""
        include_index |= index != None
        return [self.index if index == None else index]*bool(include_index) + [
            self.name,
            self.date,
            self.epoch_number,
            self.parameters.version,
            self.parameters.get_arm_selection_params()
        ] + self.parameters.training_plan[self.parameters.training_plan_index]
    
    # Load the Data From Tables
    def load_from_tables(self, parameters, trials):
        """(index), name, date, epoch_num, blocks_completed, parameter_file_version, goals, outreps, delay, goal_selection_mode, cues, forageassist, trials, success_threshold, eps_remaining"""
        
        # Load the Parameter Data
        self.index = parameters[0]
        self.name = parameters[1]
        self.date = parameters[2]
        self.epoch_number = parameters[3]
        complete_blocks = parameters[4]
        self.parameters = ParameterFile()
        self.parameters.version = parameters[6]
        self.parameters.training_plan = [parameters[7:]]
        self.parameters.training_plan_index = 0
        self.parameters._set_selected_parameters()
        
        # Load the Arm Selection Params
        for name, value in zip(('alpha', 'beta', 'gamma', 'delta'), parameters[5]):
            self.parameters.__dict__[name] = value
        
        # Load the Trial Data
        block = None
        goal = None
        for entry in trials:
            # Check if a New Goal Block has Started
            if entry[8] != goal:
                # Create a Blank Block
                block = Block(self)
                self.blocks.append(block)
                if complete_blocks:
                    block.complete = True
                    complete_blocks -= 1
            
            # Create the Trial
            trial = Trial.from_table_entry(block, entry)
            block.trials.append(trial)
            self.trials.append(trial)
        
        # Clean the Last Block
        if block: block._on_load()
    
    # Make a Raster Plot of the Epoch
    def raster_plot(self, x:int = 1, axes:plt.axes = None, back_color:str = None, alpha:float = 0.2, included:int = 0, black_line:bool = True) -> int:
        """
        Create a Raster PLot of the Epoch (either on its own if axes == None, or as a piece of a larger plot)

        Parameters
        ----------
        x : int, optional
            The x position to draw the first poke at.
            The default is 1.
        axes : plt.axes, optional
            The matplotlib axes to draw the poke on. 
            The default is None (create a new plot).
        back_color : str, optional
            The color to draw the background of the goal block.
            The default is None (white)
        alpha : float, optional
            Alters the color of the background of the plot.
            The default is 0.2.
        included : int (bitmask), optional
            Whether or not to include lockouts in the plot (included&1).
            Whether or not to include home pokes in the plot (included&2).
            The default is 0.
        black_line : bool, optional
            If true and the epoch is adding itself to a larger plot, a 
            black line will be added at the end of the epoch to note 
            when the epochs change. (this mostly exists for live-plotting 
            the data while the maze is running)
            The default is True.

        Returns
        -------
        int
            The x position of the first poke of the next block.

        """
        
        # Get the Initial Position
        x0 = x
        
        # Check if the Plot is a Subplot
        is_not_subplot = axes == None
        
        # Initialize a New Plot
        if is_not_subplot:
            fig, axes = plt.subplots()
        
        # Determine if Block-Ends should be Marked
        grey_bars = type(self.parameters.outreps) != int or self.parameters.outreps > 1
        
        # Add the Blocks
        for block in self.blocks:
            # Add the Block to the Plot
            x = block.raster_plot(x, axes, None, alpha, included)
            
            # Add a Vertical Line for the End of the Goal Block
            if grey_bars and block.complete: 
                axes.plot([x - 0.5]*2, [0, 8], color = 'lightgrey') # TODO: figure out why this is x+0.5 instead of x-0.5
        
        # Add the Background Color
        if back_color:
            axes.add_patch(plt.Rectangle(
                (0 if x0 == 1 else x0 - 0.5, 0.5),
                x - x0,
                self.rat.cohort.study.arms + 1, 
                fill = True, 
                color = back_color, 
                alpha = alpha, 
                zorder = float('inf'), 
                figure = axes.figure
            ))
        
        # Set the Display Parameters and Add Labels
        if is_not_subplot:
            fig.set_figwidth(x / 8)
            fig.set_figheight(2)
            plt.ylabel('Arm Number')
            plt.ylim(0.8 - (included&2), 6.2)
            plt.xlabel('Trial Number')
            plt.xlim(x0 - 1, x)
            plt.show()
        elif black_line:
            # Add a Vertical Line for the End of the Epoch
            axes.plot([x - 0.5]*2, [0, 8], color = 'black')
        
        # Return the Next X Position
        return x

class Rat:
    def __init__(self, name:str, data_folder:str, cohort, recursive_search:bool = True):
        # Store the Given Attributes
        self.name = name
        self.cohort = cohort
        
        # Load All the Files
        self.epochs = []
        files = search(name, data_folder, True, recursive_search)
        for filepath in files:
            if any(filepath.endswith(suffix) for suffix in Epoch._filename_parsers):
                self.epochs.append(Epoch(filepath, self))
        for filepath in (f for f in files if f.endswith('.nwb')):
            self._load_nwb(filepath)
        #if len(self.epochs) > 1: self.epochs.sort(key = lambda e: (e.date, e.epoch_number))
    
    def _load_nwb(self, filepath:str) -> None:
        pass
    
    def __repr__(self) -> str:
        return f"Rat[{self.name.capitalize()}]"
    
    def __getitem__(self, key:str):
        if type(key) == int and key > 0:
            phase_string = ''
            while key:
                phase_string += str(key&1)
                key >>= 1
            return self[phase_string]
        elif is_date(key):
            key = int(key)
            return [epoch for epoch in self.epochs if int(epoch.date) == key]
        elif is_date_range(key):
            start = int(key[:8])
            end = int(key[9:])
            return [epoch for epoch in self.epochs if start <= int(epoch.date) <= end]
        elif is_param_vector(key):
            return [epoch for epoch in self.epochs if epoch.get_param_vector() == key]
        elif is_param_vector_list(key):
            return [epoch for epoch in self.epochs if epoch.get_param_vector() in key]
        elif is_binary_string(key):
            # An Iterator for the Bits
            bits = (c == '1' for c in key)
            
            # Go Through the Epochs by Param Vector
            epochs = []
            last = None
            included = None
            for epoch in self.epochs:
                # Get the Parameter Vector
                pv = epoch.get_param_vector()
                
                # Check if the Parameter Vector has Changed
                if pv != last:
                    included = next(bits, False) # the default is necessary for when the string is too short
                    last = pv
                
                # Store the Epoch
                if included:
                    epochs.append(epoch)
            
            # Return the List of Epochs
            return epochs
        else:
            raise KeyError(f"Unrecognized Key: '{key}'")
    
    def __iter__(self):
        return iter(self.epochs)
    
    def __len__(self):
        return len(self.epochs)
    
    def __bool__(self): 
        return True # if this doesn't exist, then bool(self) = bool(len(self)) = len(self) > 0, which gives weird behavior
    
    def has_epochs(self) -> bool:
        return len(self) > 0
    
    # Make a Raster Plot of the Rat
    def raster_plot(self, axes:plt.axes = None, back_colors:list = None, alpha:float = 0.2, included:int = 0) -> int:
        """
        Create a Raster PLot for the Rat (either on its own if axes == None, or as a piece of a larger plot)

        Parameters
        ----------
        axes : plt.axes, optional
            The matplotlib axes to draw the poke on. 
            The default is None (create a new plot).
        back_colors : str, optional
            The colors to draw the background of the epochs (increments 
            through the list every time epoch.get_param_vector() changes).
            The default is None (white)
        alpha : float, optional
            Alters the color of the background of the plot.
            The default is 0.2.
        included : int (bitmask), optional
            Whether or not to include lockouts in the plot (included&1).
            Whether or not to include home pokes in the plot (included&2).
            The default is 0.

        Returns
        -------
        int
            The x position of the first poke of the next block.

        """
        
        # Check if the Plot is a Subplot
        is_not_subplot = axes == None
        
        # Initialize a New Plot
        if is_not_subplot:
            fig, axes = plt.subplots()
        
        # Add the Epochs
        x = 1
        i = 0
        back_color_assignments = {}
        for epoch in self.epochs:
            # Assign Background Colors As Needed
            param_vector = epoch.get_param_vector()
            if param_vector not in back_color_assignments:
                back_color_assignments[param_vector] = back_colors[i]
                i += 1
            
            # Add the Raster Plot for the Epoch
            x = epoch.raster_plot(x, axes, back_color_assignments[param_vector], alpha, included)
        
        # Set the Display Parameters and Add Labels
        if is_not_subplot:
            fig.set_figwidth(x / 8)
            fig.set_figheight(2)
            plt.ylabel('Arm Number')
            plt.ylim(0.8 - (included&2), 6.2)
            plt.xlabel('Trial Number')
            plt.xlim(0, x)
            plt.show()
        else:
            # Add the y-label
            axes.set_ylabel(f"{self.name.capitalize()}\nArm Number")
        
        # Return the Next X Position
        return x

class Cohort:
    def __init__(self, cohort_name:str, rat_names:list, data_folder:str, study, recursive_search:bool = True):
        self.study = study
        self.cohort_name = cohort_name
        self.rats = {name : Rat(name, data_folder, self, recursive_search) for name in rat_names}
    
    def __repr__(self) -> str:
        return f"Cohort[{self.cohort_name.capitalize()}]"
    
    def __getitem__(self, key:str):
        if type(key) == str and key in self.rats:
            return self.rats[key]
        elif type(key) == int:
            phase_string = ''
            while key:
                phase_string += str(key&1)
                key >>= 1
            return self[phase_string]
        elif is_date(key) or is_date_range(key) or is_param_vector(key) or is_param_vector_list(key) or is_binary_string(key):
            return {name : rat[key] for name, rat in self.rats.items()}
        else:
            raise KeyError(f"Unrecognized Key: '{key}'")
    
    def fetch_blocks(self, key:str, included:int = 3, complete:bool = True, filter_function = None) -> dict:
        """
        Get the Specified Epochs

        Parameters
        ----------
        key : str
            A key which helps specify which cohorts/rats/epochs to include.
        included : int, optional
            A bitmask describing:
                included&1 = include_first
                included&2 = include_others
            See get_blocks for a description of 
            indclude_first and include_others
            The default is 3.
        complete : bool, optional
            When True, only complete blocks will be returned. 
            The default is True.
        filter_function : function, optional
            An additional filter to determine which blocks to keep. 
            Must accept a Block as a parameter, and return a bool.
            The default is None (lambda block : True).
        
        Notes
        -----
        The list of blocks will always be flattened (if this is an issue, note 
        that each block has a pointer to its parent epoch: 'block.epoch')
        
        Setting included = 0 will always result in no blocks being returned
        
        Returns
        -------
        (varies depending on the key)
            key = rat name:
                returns list[Block] - the blocks for the rat which match the filter
            key = parameter vector specifier(s) or date specifiers:
                return dict[str, list[Block]] - the specified blocks, organized by rat name

        """
        
        # Process the Key
        obj = self[key]
        
        # Split the BitMask
        include_first = bool(included&1)
        include_others = bool(included&2)
        include_incomplete = not bool(complete)
        
        # Check the Result
        if isinstance(obj, dict): # {rat_name : epochs}
            # Process Each List of Epochs
            for name, epochs in obj.items():
                obj[name] = get_blocks(epochs, include_first, include_others, include_incomplete, True, filter_function)
            
            # Return the Result
            return obj
        elif isinstance(obj, Rat):
            return get_blocks(obj.epochs, included&1, included&2, not complete, True, filter_function)
        else:
            raise KeyError(f"Unrecognized Key: '{key}'")
    
    def __iter__(self):
        return iter(self.rats.values())
    
    def __len__(self):
        return len(self.rats)
    
    def __bool__(self): 
        return True # if this doesn't exist, then bool(self) = bool(len(self)) = len(self) > 0, which gives weird behavior
    
    def has_rats(self) -> bool:
        return len(self) > 0
    
    def keys(self):
        """returns a dict_keys list of rat names"""
        return self.rats.keys()
    
    def values(self):
        """returns a dict_values list of rat objects"""
        return self.rats.values()
    
    def items(self):
        """returns a dict_items list of rat_name:rat_object pairs"""
        return self.rats.items()
    
    # Make a Raster Plot for the Cohort
    def raster_plot(self, axes:plt.axes = None, back_colors:str = None, alpha:float = 0.2, included:int = 0) -> int:
        """
        Create a Raster PLot for the Cohort (either on its own if axes == None, or as a piece of a larger plot)

        Parameters
        ----------
        axes : plt.axes, optional
            The matplotlib axes to draw the poke on. 
            The default is None (create a new plot).
        back_colors : str, optional
            The colors to draw the background of the epochs (increments 
            through the list every time epoch.get_param_vector() changes).
            The default is None (white)
        alpha : float, optional
            Alters the color of the background of the plot.
            The default is 0.2.
        included : int (bitmask), optional
            Whether or not to include lockouts in the plot (included&1).
            Whether or not to include home pokes in the plot (included&2).
            The default is 0.

        Returns
        -------
        int
            The x position of the first poke of the next block.

        """
        
        # Check if the Plot is a Subplot
        is_not_subplot = axes == None
        
        # Initialize a New Plot
        if is_not_subplot:
            fig, axes = plt.subplots(len(self), 1, sharex = True, sharey = True)
        
        # Add the Rats (saving the position that's furthest to the right)
        x = max(rat.raster_plot(ax, back_colors, alpha, included) for rat, ax in zip(self.rats.values(), axes))
        
        # Set the Display Parameters and Add Labels
        if is_not_subplot:
            fig.set_figwidth(x / 8)
            fig.set_figheight(2 * len(self))
            plt.ylim(0.8 - (included&2), 6.2)
            plt.xlabel('Trial Number')
            plt.xlim(0, x)
            plt.show()
        
        # Return the Next X Position
        return x

class Study:
    def __init__(self, cohorts:dict, data_folder:str, arms:int = 6, study_name:str = None, recursive_search:bool = True, multithreaded:bool = False):
        # Store the Attributes
        self.arms = arms
        self.study_name = study_name
        self.folder = data_folder
        self.recursive_search = recursive_search
        
        # Load the Data
        if multithreaded:
            # Get the Rat Names
            self.rat_names = []
            for names in cohorts.values():
                self.rat_names.extend(names)
            
            # Initialize Blank Cohorts
            self.cohorts = {name : Cohort(name, [], None, self, recursive_search) for name, rat_names in cohorts.items()}
            
            # Map the Cohorts
            cohorts = {name : cohort for cohort, rat_names in cohorts.items() for name in rat_names}
            
            # Load All the Rats in Parallel
            with Pool() as pool:
                rats = pool.map(Study._load_rat, [(name, data_folder, cohorts[name], recursive_search) for name in self.rat_names])
            
            # Add the Rats to their Cohorts
            for rat in rats:
                self.cohorts[rat.cohort][rat.name] = rat
        else:
            # Load the Data
            self.cohorts = {name : Cohort(name, rat_names, data_folder, self, recursive_search) for name, rat_names in cohorts.items()}
        
        # Create a "Cohort" which Contains All the Rats (note: this does not re-load the data)
        self.rat_names = []
        self.cohorts['all'] = Cohort('all', [], None, self, False)
        for name, cohort in self.cohorts.items():
            if name != 'all':
                for name, rat in cohort.rats.items():
                    self.cohorts['all'].rats[name] = rat
                    self.rat_names.append(name)
    
    def _load_rat(name_folder_cohort_rsearch:tuple) -> Rat:
        return Rat(*name_folder_cohort_rsearch)
    
    def __repr__(self) -> str:
        if self.study_name:
            return f"Study[{self.study_name}]"
        else:
            names = str([name.capitalize() for name in self.rat_names]).replace("'", '')
            return f"Study{names}"
    
    def __getitem__(self, key:str):
        if type(key) == str and key in self.cohorts:
            return self.cohorts[key]
        elif type(key) == str and key in self.cohorts['all'].rats:
            return self.cohorts['all'].rats[key]
        elif type(key) == int:
            phase_string = ''
            while key:
                phase_string += str(key&1)
                key >>= 1
            return self[phase_string]
        elif is_date(key) or is_date_range(key) or is_param_vector(key) or is_param_vector_list(key) or is_binary_string(key):
            return {name : cohort[key] for name, cohort in self.cohorts.items() if name != 'all'}
        else:
            raise KeyError(f"Unrecognized Key: '{key}'")
    
    def fetch_blocks(self, key:str, included:int = 3, complete:bool = True, filter_function = None) -> dict:
        """
        Get the Specified Epochs

        Parameters
        ----------
        key : str
            A key which helps specify which cohorts/rats/epochs to include.
        included : int, optional
            A bitmask describing:
                included&1 = include_first
                included&2 = include_others
            See get_blocks for a description of 
            indclude_first and include_others
            The default is 3.
        complete : bool, optional
            When True, only complete blocks will be returned. 
            The default is True.
        filter_function : function, optional
            An additional filter to determine which blocks to keep. 
            Must accept a Block as a parameter, and return a bool.
            The default is None (lambda block : True).
        
        Notes
        -----
        The list of blocks will always be flattened (if this is an issue, note 
        that each block has a pointer to its parent epoch: 'block.epoch')
        
        Setting included = 0 will always result in no blocks being returned

        Returns
        -------
        (varies depending on the key)
            key = rat name:
                returns list[Block] - the blocks for the rat which match the filter
            key = cohort hame:
                returns dict[str, dict[str, list[Block]]] - the requested blocks 
                organized by cohort, and then by rat name
            key = parameter vector specifier(s) or date specifiers:
                return dict[str, list[Block]] - the specified blocks, organized by rat name

        """
        
        # Process the Key
        obj = self[key]
        
        # Check the Result
        if isinstance(obj, dict): # {cohort_name : {rat_name : epochs}}
            # Process Each List of Epochs
            for cohort in obj.values():
                for name, epochs in cohort.items():
                    cohort[name] = get_blocks(epochs, included&1, included&2, not complete, True, filter_function)
            
            # Return the Result
            return obj
        elif isinstance(obj, Cohort):
            # Build a Dictionary {rat_name : blocks}
            return {name : get_blocks(rat.epochs, included&1, included&2, not complete, True, filter_function) for name, rat in obj.items()}
        elif isinstance(obj, Rat):
            return get_blocks(obj.epochs, included&1, included&2, not complete, True, filter_function)
        else:
            raise KeyError(f"Unrecognized Key: '{key}'")
    
    def __iter__(self):
        return iter(self.cohorts.values())
    
    def __len__(self):
        return len(self.cohorts)
    
    def __bool__(self): 
        return True # if this doesn't exist, then bool(self) = bool(len(self)) = len(self) > 0, which gives weird behavior
    
    def has_cohorts(self) -> bool:
        return len(self) > 0
    
    def keys(self):
        """returns a dict_keys list of cohort names"""
        return self.cohorts.keys()
    
    def values(self):
        """returns a dict_values list of cohort objects"""
        return self.cohorts.values()
    
    def items(self):
        """returns a dict_items list of cohort_name:cohort_object pairs"""
        return self.cohorts.items()
    
    # Make a Raster Plot for the Cohort
    def raster_plot(self, cohort:str = 'all', back_colors:str = None, alpha:float = 0.2, included:int = 0) -> int:
        """
        Create a Raster PLot for the Cohort (either on its own if axes == None, or as a piece of a larger plot)

        Parameters
        ----------
        cohort : str or list, optional
            The cohort to plot, or a list of rat names to plot
        back_colors : str, optional
            The colors to draw the background of the epochs (increments 
            through the list every time epoch.get_param_vector() changes).
            The default is None (white)
        alpha : float, optional
            Alters the color of the background of the plot.
            The default is 0.2.
        included : int (bitmask), optional
            Whether or not to include lockouts in the plot (included&1).
            Whether or not to include home pokes in the plot (included&2).
            The default is 0.
        """
        if type(cohort) == str:
            self.cohorts[cohort].raster_plot(1, None, back_colors, alpha, included)
        elif type(cohort) == list:
            temp_cohort = Cohort('temp', [], None, self, False)
            for rat in cohort:
                temp_cohort.rats[rat] = self[rat]
            temp_cohort.raster_plot(1, None, back_colors, alpha, included)
        else:
            raise TypeError(f"<cohort> must either be the name of a cohort (str), or a list of rat names (list), not type '{type(cohort)}'")

"""
Helpers
"""

def is_date(s:str) -> bool: 
    return (type(s) == str and len(s) == 8 and s.isdigit()) or (type(s) == int and len(str(s)) == 8)

def is_date_range(s:str) -> bool:
    return type(s) == str and len(s) == 17 and s[8] == '-' and s[:8].isdigit() and s[9:].isdigit()

def is_param_vector(s:tuple) -> bool:
    return type(s) == tuple and not Epoch.param_vector_attributes or len(s) == len(Epoch.param_vector_attributes)

def is_param_vector_list(s:list) -> bool:
    return type(s) == list and all(is_param_vector(x) for x in s)

def is_binary_string(s:str) -> bool:
    return type(s) == str and all(c in '01' for c in s)

def get_blocks(epochs:list, include_first:bool = True, include_others:bool = True, include_incomplete:bool = False, flatten:bool = True, filter_function = None) -> list:
    """
    Extract a List of Blocks from the List of Epochs Using the Specified Filters

    Parameters
    ----------
    epochs : list[Epoch]
        A list of epochs to extract blocks from.
    include_first : bool, optional
        When False, the first block of every 
        epoch will be ignored. 
        The default is True.
    include_others : bool, optional
        When False, the only blocks which will be 
        included in the output will be the first block 
        of every epoch (unless include_first == False, then 
        the output will be an empty list, since nothing will 
        have been selected to be included). 
        The default is True.
    include_incomplete : bool, optional
        When False, incomplete blocks are excluded from 
        the output. The default is False.
    flatten : bool, optional
        When True the blocks are all returned in a single list, when 
        False, the blocks are returned as a list[list[Block]], where 
        the blocks have been grouped by epoch. 
        The default is True.
    filter_function : function, optional
        An additional filter to determine which blocks to keep. 
        Must accept a Block as a parameter, and return a bool.
        The default is None (lambda block : True).
    
    Notes
    -----
    If include_first == include_others == False, this method will 
    always return an empty list, without complaining about the input.
    
    Returns
    -------
    list[Block] or list[list[Block]]
        A list of the requested blocks (grouped by epoch if flatten == False).

    """
    
    # Initialize a List of Blocks, and a Pointer to the List to Append To
    blocks = []
    append_to = blocks
    
    # Initialize a Dummy Filter Function
    if filter_function == None:
        filter_function = lambda block : True
    
    # Get Blocks from Every Epoch in the List
    for epoch in epochs:
        # Start a New Group (if flatten == False)
        if not flatten:
            blocks.append([])
            append_to = blocks[-1]
        
        # Iterate Over the Range of Blocks to Check
        for i in range(1 - include_first, (len(epoch.blocks) if include_others else 2)):
            # Get the Block
            block = epoch.blocks[i]
            
            # Check if it Should be Included
            if (block.complete or include_incomplete) and filter_function(block):
                # Add the Block to the List
                append_to.append(block)
    
    # Return the List of Blocks
    return blocks