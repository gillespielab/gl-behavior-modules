# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:22:12 2025

-----------------------
PARAMTER FILE INTERFACE
-----------------------

This is essentially a plug-in for environments 
(currently mostly radial mazes) which allows the 
environment to have persistent memory to do things 
like storing parameters, recording stats, and 
scheduling a training curriculum.
 

-----------------------------------
ADDING A NEW PARAMETER FILE VERSION
-----------------------------------

To add a version you must add a Parser which can parse a single line of the 
training plan, as wells as a Parser which uses the first parser to parse the 
entire file. You must also specify the attribute names to assign the parsed 
values to. 

Currently, there are 5+1 items you need to update, and they're all 
flagged with the comment:
    #  TODO: update this when adding versions
Search for this comment, and then mimic what you see, and you should be fine!


Note that for the training plan to work, the last 2 values in each row must 
be the epoch counts (successful_epochs_remaining, total_epochs_remaining). 
The ParameterFile object expects those parameters to be in the last 2 
positions of each row in the training plan, and when selecting which row 
to use it skips all rows that have a 0 for each count.

Also, you will generally have more success with objects built using 
ParameterFile objects if the attribute names you select match the 
attribute names already in use


------------
DEPENDENCIES
------------

Modules.TextParser
Modules.OSTools
Modules.StateScriptInterface (optional)


@author: Violet Saathoff
"""

# Import Libraries
try:
    from Modules.StateScriptInterface import disp
    from Modules.TextParser import Parser
    from Modules.Utils import readlines, writelines
except:
    try:
        from StateScriptInterface import disp
    except:
        def disp(message:str, log_message:bool = False):
            """Placeholder for StateScriptInterface.disp
            
            Notes
            -----
            log_message does nothing
            messages are not sanitized
            """
            print(f"SCQTMESSAGE: disp('{message}');")
    from TextParser import Parser
    from Utils import readlines, writelines

"""
Misc Parsers
"""

version_parser = Parser("Version {int}")

"""
Parameter Line Parsers
"""

# A Function to Parse the Various Options for OutReps
def parse_outreps(s:str):
    """A Function for Parsing a String as a Value for Selecting outreps
    
    Behavior After Casting Values to Integers:
        - if s is an integer: outreps = s
        - if s is a 2-tuple: outreps = np.randint(*s)
        - if s is a list: outreps = np.random.choice(s)
    
    Raises a ValueError if <s> doesn't match any of these formats
    
    Added to Parser.datatypes with the key 'outreps'
    """
    s = s.replace(' ', '')
    if s.isdigit() or (s[0] == '-' and s[1:].isdigit()):
        return int(s)
    elif s[0] == '(' and s[-1] == ')':
        return tuple(map(int, s[1:-1].split(',')))
    elif s[0] == '[' and s[-1] == ']':
        return list(map(int, s[1:-1].split(','))) if len(s) > 2 else []
    else:
        raise ValueError(f"unrecognized outreps: '{s}'")
Parser.add_datatype('outreps', parse_outreps, r'(\+?\d+|\(\+?\d+,\s*\+?\d+\)|\[\d+(,\s*\d+)*\])')

# A Function to Parse the Various Options for Success Threshold
def parse_threshold(s:str):
    """parse <s> as an int if possible, or as a float if not
    
    Raises a ValueError if <s> cannot be parsed as either an 
    integer or a float.
    
    Added to Parser.datatypes with the key 'threshold'
    """
    return int(s) if s.isdigit() else float(s)
Parser.add_datatype('threshold', parse_threshold, Parser.regex_components[float])

# A function to Parse an Integer or a Tuple
def parse_epoch_end_criteria(s:str):
    """parse <s> as an integer if possible, or as a 2-tuple 
    of integers otherwise"""
    s = s.replace(' ', '')
    if s.isdigit() or (s[0] == '-' and s[1:].isdigit()):
        return int(s)
    elif s[0] == '(' and s[-1] == ')':
        return tuple(map(int, s[1:-1].split(',')))
    else:
        raise ValueError(f"unrecognized epoch end criteria: '{s}'")
Parser.add_datatype('ep_end', parse_epoch_end_criteria, r'(\+?\d+|\(\+?\d+,\s*\+?\d+\))')

# A function to Parse a Nullable List of Integers
def parse_nullable_intlist(s:str):
    """Parse a String as None or as a List of Integers"""
    if not s or s == 'None':
        return None
    elif s[0] == '[' and s[-1] == ']':
        s = s.replace(' ', '')
        return list(map(int, s[1:-1].split(','))) if len(s) > 2 else []
    else:
        raise ValueError(f"'{s}' cannot be interpreted as a nullable list of integers")
Parser.add_datatype('nullable_intlist', parse_nullable_intlist, r'(None|\[\d+(,\s*\d+)*\])')


# parameter template strings
parameters = [
    None, # V0 doesn't exist
    "{int} {outreps} {int} {int} {int} {int} {int} {int} {float} {int} {int}",
    "{int} {outreps} {int} {int} {int} {int} {int} {int} {threshold} {int} {int}",
    "{int} {outreps} {int} {int} {int} {int} {ep_end} {threshold} {int} {int} {int}"
] #  TODO: update this when adding versions

# The Names of the Attributes in the Loaded ParameterFile Object (it's important that these names stay consistent from version to version)
parameter_attribute_names = [
    None, # V0 doesn't exist
    ["goals", "outreps", "delay", "goal_selection_mode", "cues", "forageassist", "min_trials", "max_trials", "success_threshold", "successful_epochs_remaining", "max_epochs_remaining"],
    ["goals", "outreps", "delay", "goal_selection_mode", "cues", "forageassist", "min_trials", "max_trials", "success_threshold", "successful_epochs_remaining", "max_epochs_remaining"],
    ["goals", "outreps", "delay", "goal_selection_mode", "cues", "forageassist", "epoch_end_criteria", "success_threshold", "timeout","successful_epochs_remaining", "max_epochs_remaining"]
] #  TODO: update this when adding versions

# parameter parsers
parameter_parsers = [Parser(params, list) if params else None for params in parameters]

# Add the Parameter Parsers as 'datatypes'
for i in range(1, len(parameter_parsers)):
    Parser.add_datatype(f"v{i}_params", parameter_parsers[i])
del(i)


"""
File Parsers"""
v1_pattern = """Version 1

Current State:
arm visits (rewarded): [{[int, ',']}]
arm visits (total): [{[int, ',']}]
goal selections:  [{[int, ',']}]
goal sequence: [{[int, ',']}]

Training Plan: (numgoals, outreps, locktime, arm selection mode, cued, forageassist, min trials, max trials, success rate, successful epochs, max epochs)
{[v1_params, '\n']}"""

v2_pattern = """Version 2

Current State:
arm visits (rewarded): [{[int, ',']}]
arm visits (total): [{[int, ',']}]
arm visits (weighted): [{[float, ',']}]
goal selections:  [{[int, ',']}]
goal sequence: [{[int, ',']}]
arm selection parameters (alpha, beta, gamma): ({float}, {float}, {float})
epoch timeout (minutes): {int}

Training Plan: (numgoals, outreps, locktime, arm selection mode, cues, forageassist, min trials, max trials, success threshold, successful epochs, max epochs)
{[v2_params, '\n']}"""

v3_pattern = """Version 3

Current State:
arm visits (rewarded): [{[int, ',']}]
arm visits (total): [{[int, ',']}]
arm visits (weighted): [{[float, ',']}]
goal selections:  [{[int, ',']}]
goal sequence: [{[int, ',']}]
arm selection parameters (alpha, beta, gamma, delta): ({float}, {float}, {float}, {float})
last goal: {nullable_intlist}

Training Plan: (numgoals, outreps, locktime, arm selection mode, cues, forageassist, epoch end criteria, success threshold, timeout successful epochs, max epochs)
{[v3_params, '\n']}"""

patterns = [
    None, # V0 doesn't exist
    v1_pattern,
    v2_pattern,
    v3_pattern
] #  TODO: update this when adding versions

# The Names of the Attributes in the Loaded ParameterFile Object (it's important that these names stay consistent from version to version)
attribute_names = [
    None, # V0 doesn't exist
    ["rewarded_visits", "total_visits", "goal_counts", "goal_sequence", "training_plan"],
    ["rewarded_visits", "total_visits", "weighted_visits", "goal_counts", "goal_sequence", "alpha", "beta", "gamma", "timeout", "training_plan"],
    ["rewarded_visits", "total_visits", "weighted_visits", "goal_counts", "goal_sequence", "alpha", "beta", "gamma", "delta", "last_goal", "training_plan"]
] #  TODO: update this when adding versions

# Make the File Parsers
parsers = [Parser(pattern) if pattern else None for pattern in patterns]


"""
Parameter File Object
"""

class ParameterFile:
    def __init__(self, filepath:str = None, version:int = -1):
        """
        Load a Parameter File, or Create a Blank Parameter File Object

        Parameters
        ----------
        filepath : str, optional
            The Filepath to the Parameter File. 
            The default is None (creates a blank 
            object with null data).
        version : int, optional
            The version of the ParameterFile to 
            create if filepath == None (otherwise 
            the version is set automatically on 
            reading the parameter file).
            The default is -1 (the most recent version).

        """
        
        # Save the Filepath
        self.version = version if version >= 0 else len(attribute_names) + version
        self.filepath = filepath
        
        # Special Indices
        self.outreps_index = 1 #  TODO: update this when adding versions (if necessary)
        self.epoch_end_criteria_index = 6 if self.version == 3 else None #  TODO: update this when adding versions (if necessary)
        
        # Load the File
        self.min_trials = -1
        self.max_trials = -1
        self.goal_blocks = -1
        self.end_mode = 0
        if self.filepath:
            self.load_from_file()
        else:
            # Generate Blank Attributes
            for name in attribute_names[self.version]:
                self.__dict__[name] = None
            for name in parameter_attribute_names[self.version]:
                self.__dict__[name] = None
    
    def load_from_file(self, lines:list = None):
        """Load the Values in the Parameter File Onto the ParameterFile Object"""
        # Read the File
        self.parameter_file_lines = lines if lines else readlines(self.filepath)
        
        # Parse the Version
        self.version = version_parser(self.parameter_file_lines[0])
        
        # Re-Build the File
        file = '\n'.join(self.parameter_file_lines)
        
        # Parse the File
        values = parsers[self.version](file)
        
        # Store the Values in the Appropriate Attributes
        for name, val in zip(attribute_names[self.version], values):
            self.__dict__[name] = val
        
        # Find the Training Plan Index
        self.training_plan_index = 0
        while self.training_plan_index < len(self.training_plan) and 0 in self.training_plan[self.training_plan_index][-2:]:
            self.training_plan_index += 1
        
        # Check if No Training Plan Index was Found
        if self.training_plan_index == len(self.training_plan):
            # Print a Warning
            print('Warning: no parameter set with epochs remaining found, using parameters from the last line of the training plan')
            
            # Use the Last Row
            self.training_plan_index -= 1
        
        self._set_selected_parameters()
        
    def _set_selected_parameters(self):
        """Assign the Attributes Associated with the Training Plan"""
        # Get the Training Plan Values
        for name, val in zip(parameter_attribute_names[self.version], self.training_plan[self.training_plan_index]):
            self.__dict__[name] = val
        
        # Get the Epoch End Criteria
        if self.version == 3:
            if type(self.epoch_end_criteria) == int:
                self.goal_blocks = self.epoch_end_criteria
                self.end_mode = 1
            elif type(self.epoch_end_criteria) == tuple and len(self.epoch_end_criteria) == 2:
                self.min_trials, self.max_trials = self.epoch_end_criteria
                self.end_mode = 0
            else:
                raise ValueError(f"Unrecognized Epoch End Criteria: '{self.epoch_end_criteria}'")
    
    def get_parameter_file_lines(self) -> list[str]:
        """Get the Updated Lines of the Parameter File (mostly used when saving the file)"""
        # Update Values Known to Change
        self.training_plan[self.training_plan_index][-2] = self.successful_epochs_remaining
        self.training_plan[self.training_plan_index][-1] = self.max_epochs_remaining
        
        # Get the Header Values
        values = [self.__dict__[name] for name in attribute_names[self.version]]
        
        # Edit the Outreps/Epoch End Criteria
        for row in values[-1]:
            for index in [self.outreps_index, self.epoch_end_criteria_index]:
                if index != None:
                    row[index] = str(row[index]).replace(' ', '')
        
        # Build/Return the Lines
        return parsers[self.version].build(values).split('\n')
        
    
    def save_to_file(self):
        """Update the Values in the Parameter File with the Values Currently in the ParameterFile Object"""
        writelines(self.filepath, self.get_parameter_file_lines())
    
    def get_param_string(self):
        """Generate a String Containing the Parameters from the Selected Training Plan Line"""
        end_criteria = str(self.epoch_end_criteria).replace('(', '[').replace(')', ']') if self.version == 3 else ''
        return [
            f"params: [goals: {self.goals}, outreps: {self.outreps}, delay: {self.delay}, cued: {self.cues}, forageassist: {self.forageassist}, trials: [{self.min_trials}, {self.max_trials}], timeout: {self.timeout}, success_threshold: {self.success_threshold}, epochs_remaining: [{self.successful_epochs_remaining}, {self.max_epochs_remaining}]]",
            f"params: [goals: {self.goals}, outreps: {self.outreps}, delay: {self.delay}, cued: {self.cues}, trials: [{self.min_trials}, {self.max_trials}], success_threshold: {self.success_threshold}, epochs_remaining: [{self.successful_epochs_remaining}, {self.max_epochs_remaining}]]",
            f"params: [goals: {self.goals}, outreps: {self.outreps}, delay: {self.delay}, arm selection mode: {self.goal_selection_mode}, cues: {self.cues}, forageassist: {self.forageassist}, trials: [{self.min_trials}, {self.max_trials}], success_threshold: {self.success_threshold}, epochs_remaining: [{self.successful_epochs_remaining}, {self.max_epochs_remaining}], timeout: {self.timeout}]",
            f"params: [goals: {self.goals}, outreps: {self.outreps}, delay: {self.delay}, arm selection mode: {self.goal_selection_mode}, cues: {self.cues}, forageassist: {self.forageassist}, end criteria: {end_criteria}, success_threshold: {self.success_threshold}, epochs_remaining: [{self.successful_epochs_remaining}, {self.max_epochs_remaining}], timeout: {self.timeout}]"
        ][self.version] #  TODO: update this when adding versions
    
    def print_params(self):
        """Print a Selection of Important Parameters from the Parameter File"""
        # Print the Maze Parameters
        if self.version == 1: #  TODO: update this when adding versions (as needed)
            for line in [
                f"rewarded visits: {list(self.rewarded_visits)}",
                f"total visits:         {list(self.total_visits)}",
                f"goal selections: {self.goal_counts}",
                self.get_param_string()
            ]: disp(line)
        else:
            for line in [
                f"rewarded visits: {list(self.rewarded_visits)}",
                f"total visits:         {list(self.total_visits)}",
                f"weighted visits: {[round(x, 3) for x in self.weighted_visits]}",
                f"goal selections: {self.goal_counts}",
                self.get_param_string()
            ]: disp(line)
    
    def get_arm_selection_params(self):
        """Get the Arm Selection Parameters by Version"""
        if self.version == 1:
            return (self.alpha, self.beta)
        elif self.version == 2:
            return (self.alpha, self.beta, self.gamma)
        else:
            return (self.alpha, self.beta, self.gamma, self.delta)

class Sentinel:
    """An Empty Class to be Used as a Sentinal in Certain Applications"""
