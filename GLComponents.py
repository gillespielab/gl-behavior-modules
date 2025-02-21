# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:34:28 2025

@author: Violet
"""

# Import Libraries
import time
import numpy as np
from numpy.random import randint, choice, shuffle
from collections import defaultdict
try:
    from Modules import StateScriptInterface as ssi
    from Modules.ParameterFileInterface import ParameterFile
except:
    import StateScriptInterface as ssi
    from ParameterFileInterface import ParameterFile


"""
Enums
"""

class port_groups:
    """A Pseudo-Enum for Storing the Port Groups Names"""
    pump = 'Pump'
    well = 'Well'

class well_states:
    """An Enum Containing the Reward Well States"""
    inactive = 0
    active = 1
    reward_given = 2


"""
Reward Wells
"""

class Pump(ssi.Port):
    """An Object which Manages Delivering Reward via a Pump (auto-generated when creating Wells)"""
    
    updates_enabled_default = False
    
    def __init__(self, port:int, reward_function:int, rewarded_pump_var_name:str = None, updates_off:bool = None):
        """
        Parameters
        ----------
        port : int
            The Port Number of the Pump (as used by the ECU).
        reward_function : int
            The Function Number (in StateScript) to trigger to Deliver a Reward.
        rewarded_pump_var_name : str, optional
            The name of the variable in StateScript which stores the 
            port number for the active pump. 
            The default is wells.default_reward_pump_var_name.
        updates_off : bool, optional
            When True, DIO updates for this pump will be disabled. (this can clean up the StateScript, 
            but makes it a little harder to detect if there are hardware issues)
            The default is Pump.updates_enabled_default
            
        """
        # Initialize the Port
        updates_off = ssi._null_coalescing(updates_off, not Pump.updates_enabled_default)
        super(Pump, self).__init__(port, port_groups.pump, dio_updates = not updates_off) #  calls Port.__init__()
        
        # Initialize Pump-Specific Attributes
        self.reward_function = ssi.functions[reward_function] if reward_function in ssi.functions else reward_function
        self.rewarded_pump_var_name = ssi._null_coalescing(rewarded_pump_var_name, wells.default_reward_pump_var_name)
    
    def deliver_reward(self):
        """Deliver a Reward Using the Stored Parameters"""
        ssi.set_var(self.rewarded_pump_var_name, self.port)
        ssi.trigger(self.reward_function)
    
    def __repr__(self):
        return f"<Pump {self.index}>"


class _wells(list):
    """A List of All Well Objects, Coupled With a Dictionary of Well Objects by Group"""
    
    def __init__(self, well_list:list = (), append_only = True):
        """Create a List of Wells/Dictionary of Grouped Wells"""
        super(_wells, self).__init__(well_list)
        self.groups = defaultdict(list)
        self.append_only = append_only
        
        # convenience variable
        self.default_reward_pump_var_name = 'rewardPump'
    
    def __contains__(self, key) -> bool:
        """Check if a Group Name or Well is in the List"""
        return (type(key) in (str, int, tuple) and key in self.groups) or super(_wells, self).__contains__(key)
    
    def append(self, well):
        """Append a Well to the List of Wells (and to the grouped well dictionary)"""
        super(_wells, self).append(well)
        self.groups[well.group].append(well)
    
    def pop(self, index:int = -1):
        """Remove and Return the Well at the Specified Index (be careful, this could break the well indices)"""
        if not self.append_only:
            well = super(_wells, self).pop(index)
            self.groups[well.group].remove(well)
        elif ssi.config.verbose:
            ssi.disp("Well cannot be removed from well list because the well list is append only")
    
    def remove(self, well):
        """Remove a Well from the List (be careful, this could break the well indices)"""
        if not self.append_only:
            super(_wells, self).remove(well)
            self.groups[well.group].remove(well)
        elif ssi.config.verbose:
            ssi.disp("Well cannot be removed from well list because the well list is append only")
    
    def __getitem__(self, key):
        """Get a Well(s) by Index or Group Name 
        
        This checks if the key maps to a group first, so using integers as 
        group names can lead to collisions/unexpected behavior. I recommend 
        not using integers as group names, but if you really want to do that 
        then I recommend using _wells.get_group(self, group:str) or 
        _wells.get_well(self, index:int) to avoid unexpected behavior.
        """
        if key in self.groups:
            return self.groups[key]
        else:
            return super(_wells, self).__getitem__(key)
    
    def __setitem__(self, *args, **kwargs):
        """Setting Items in this List is Forbidden"""
        if ssi.config.verbose: ssi.disp('Setting Items in the Well List is Forbidden')
    
    def get_group(self, group:str):
        """Get the List of All Wells in the Specified Group (only necessary if group names are integers)"""
        return self.groups[group]
    
    def get_well(self, index:int):
        """Get a Well by its Index (only necessary if group names are integers)"""
        return super(_wells, self).__getitem__(index)
    
    def clear(self):
        """Remove All the Wells from the List"""
        super(_wells, self).clear()
        self.groups.clear()
    
    def _get_well_list(self, well_list) -> list:
        if well_list == None:
            return self
        elif type(well_list) == Well:
            return [well_list]
        elif type(well_list) in (str, int, tuple):
            return self.groups[well_list]
        else:
            return well_list
    
    def activate(self, well_list:list = None, set_leds:bool = True, update:bool = False):
        """Activate All Wells in the Given List/Group"""
        # Activate the Wells
        for well in self._get_well_list(well_list):
            well.activate()
        
        # Set the LEDs
        if set_leds:
            for well in self._get_well_list(well_list):
                well.led.on()
        
        # Update the Wells
        if update: self.update(well_list)
    
    def deactivate(self, well_list:list = None, set_leds:bool = True, update:bool = False):
        """Deactivate All Wells in the Given List/Group"""
        # Get the Wells List
        well_list = self._get_well_list(well_list)
        
        # Activate the Wells
        for well in well_list:
            well.deactivate()
        
        # Set the LEDs
        if set_leds:
            for well in well_list:
                well.led.off()
        
        # Update the Wells
        if update: self.update(well_list)
    
    def set_leds(self, well_list:list = None, state:int = 1):
        """Set All the LEDs in the Given List/Group"""
        for well in self._get_well_list(well_list):
            well.set_output(state)
    
    def update(self, well_list = None):
        """Set the LEDs to their next output states (if set)"""
        for well in self._get_well_list(well_list):
            well.update_output()
    
    def configure(self, well_ports:list, pump_ports:list, port_params:dict):
        """
        Create the Well Objects Specified by the <ports> dictionary

        Parameters
        ----------
        well_ports : list
            A list of all the ports to use for the wells. E.g.
                well_ports = [7, 1, 2, 3, 4, 5, 6]
        pump_ports : list
            A list of all the ports to use for the pumps. E.g.
                pump_ports = [15, 9, 10, 11, 12, 13, 14]
        port_params : dict
            A mapping of group names to well parameters and indices. E.g.
                port_params = {
                    'home': ((1, 50), [0]), # cued = 1, reward_function = 50uL
                    'arms': ((0, 150), [1, 2, 3, 4, 5, 6]) # cued = 0, reward_function = 150uL
                }
        
        Together, the example parameters create a 6-arm maze with 1 home 
        well (ports 7 and 15), and 6 arm wells (ports 1-6 for the wells, 
        and ports 9-14 for the pumps). Note that the indices in <port_params> 
        are used to access the port numbers in both <well_ports> and <pump_ports>.
        Example Call:
            
            ssi.configure(1, {50:10, 100:11, 150:12}) #  setting the reward functions must be done first if you're using a reward function mapping (this also sets the stats function to 1)
            wells.configure(
                [7, 1, 2, 3, 4, 5, 6],
                [15, 9, 10, 11, 12, 13, 14],
                {
                    'home': ((1, 50), 1), # replacing "[0]" with "1" means that it takes the next 1 indices (0)
                    'arms': ((0, 150), 6) # replacing "[1, 2, 3, 4, 5, 6]" with "6" means that it takes the next 6 indices (1-6)
                }
            )

        """
        used_indices = 0
        next_index = 0
        for group, (params, indices) in port_params.items():
            # Build the List of Indices (if necessary)
            if type(indices) == int:
                n = indices
                indices = []
                while len(indices) < n:
                    if not used_indices & (1 << next_index):
                        indices.append(next_index)
                    next_index += 1
            
            # Create the Well Objects
            for i in indices:
                used_indices |= 1 << i
                Well(well_ports[i], pump_ports[i], group, *params, active_well_list = self)
    
    def forbid_simultaneous_pokes(self, well_list:list = None) -> None:
        """Prints an Error Message if 2+ IR Beams are Tripped at the Same Time"""
        if well_list == None:
            ssi.config.forbid_simultaneous_inputs(port_groups.well)
        else:
            port_list = [well for well in self._get_well_list(well_list)]
            ssi.config.forbid_simultaneous_inputs(port_list)
    
    def forbid_simultaneous_rewards(self, well_list:list = None) -> None:
        """Prints an Error Message and Stops Reward Delivery if 2+ Pumps are Running at the Same Time"""
        if well_list == None:
            ssi.config.forbid_simultaneous_outputs(port_groups.pump)
        else:
            port_list = [well.pump for well in self._get_well_list(well_list)]
            ssi.config.forbid_simultaneous_outputs(port_list)

# Create an Alias for the Well List
WellList = _wells

# Create the Default Well List
wells = _wells()


class Well(ssi.Port):
    """An Object which Manages Reward Wells (controlling the LED, the IR Beam, and the Pump)
    
    Global Attributes
    -----------------
    reward_cooldown : float (seconds)
        Setting this to a value >0 will assert that a reward well cannot deliver a reward 
        if any other reward well delivered a reward <reward_cooldown seconds ago.
        The default is -1 (disabled)
    last_reward : float (seconds)
        The last time any well delivered a reward (set using time.time())
    
    Well Attributes
    ---------------
    group : str
        The name of the group the reward well belongs to (e.g. 'home', 'arms')
    led : Well
        led = self - is an alias of the current object to help make code more 
        readable. E.g. it lets you write "well.led.on()" rather than "well.on()"
    pump : Pump
        The Pump object which controls the pump that delivers reward to the well.
    cued : bool
        Whether or not the well is cued. Wells that are cued will update their 
        LED state on well.activate() and well.deactivate()
    state : int
        0 - inactive (the well will not give reward)
        1 - active (the well will give reward)
        2 - the well was active, and gave a reward (the well will not give a reward)
    reward_given : bool
        Whether or not the well has given a reward since the last time the well was activated.
        (only resets on well.activate())
    pokes : int
        The number of times the well has been poked (this epoch)
    rewards : int
        The number of rewards the well has delivered (this epoch)
    delay_outputs : bool
        When True, the output of the DIO port is only updated on well.update().
        (This is actually inherited from StateScriptInterface.Port) - The default is True.
    
    Port Attributes
    ---------------
    port : int
        The port number (as recornized by the ECU)
    port_group : str
        The group of objects the port belongs to (E.g. 'reward_wells')
    index : int
        The index of the Port object in the list of Port objects which 
        share the same group (the index is not unique between all ports 
        unless all Port objects share the same group)
    bitmask : int
        A filter for interpreting the input/output state of the port from 
        state updates from StateScript. (i.e. when StateScript outputs 3 
        integers, the first integer is a timestamp (like usual), the second 
        integer is a bitmask representing which inputs are high, and the 
        third integer is a bitmask representing which outputs are high).
        The value of this bitmask should always be 2**(port - 1), or 
        equivalently: 1 << (port - 1). DO NOT EDIT THIS VALUE
    input_val : int
        The current input value of the port (only udpates itself when the 
        callback function is built around StateScriptInterface._callback)
    output_val : int
        The current output value of the port (only udpates itself when the 
        callback function is built around StateScriptInterface._callback. 
        Otherwise it assumes the output is initialized to 0, and then tracks 
        when the value is updated in Python - this can be broken if the 
        output value is set directly from StateScript)
    next_output_val : int
        The value to update the output to the next time update_output() is 
        called. (only used if delay_outputs == True)
    
    Notes
    -----
    Most reward well functionality is done more easilly with 'wells' (a default instance of class _wells).
    
    """
    
    # can be used as a safety check to make sure rewards aren't being given impossibly fast
    # which could be a sign of broken hardware
    last_reward = 0
    reward_cooldown = -1
    
    def __init__(self, well_port:int, pump_port:int, group:str, cued:bool, reward_function:int, rewarded_pump_var_name:str = None, active_well_list:_wells = wells):
        """I recommend initializing wells with wells.configure() rather than making each well on its own
        
        Parameters
        ----------
        well_port : int
            The Port Number of the IR Beam/LED (as used by the ECU).
        pump_port : int
            The Port Number of the Pump (as used by the ECU).
        cued : bool
            Whether or not the LED sate should be tied to whether or not the well is active.
        group : str
            The group of wells this well belongs to (it is recommended to use strings, but 
            <group> can be any hashable object).
        reward_function : int
            The Function Number (in StateScript) to trigger to Deliver a Reward.
        rewarded_pump_var_name : str, optional
            The name of the variable in StateScript which stores the 
            port number for the active pump. The default is Well.default_reward_pump_var_name.
        active_well_list : _wells, optional
            MODIFY THIS WITH CAUTION. Overriding this can allow you to have multiple 
            separate well lists (e.g. if you're controlling multiple environments 
            simultaneously from a single computer/ECU), but then it is on you to 
            manage those well lists.
        """
        
        # Set the Parameters
        super(Well, self).__init__(well_port, port_groups.well, delayed_outputs = True)
        self.group = group
        self.led = self # alias which can maybe make code more readable? (e.g. "well.led.on()" is more clear than "well.on()"?)
        self.pump = Pump(pump_port, reward_function, rewarded_pump_var_name)
        self.cued = cued
        self.state = well_states.inactive
        self.pokes = 0
        self.rewards = 0
        self.reward_given = False
        
        # Add the Well to the List of Wells
        active_well_list.append(self)
        
        # Log that the Well Was Initialized
        ssi.log(f'Well {self.index} Initialized: [group = {self.group}, led_port = {self.port}, pump_port = {self.pump.port}, cued = {self.cued}, reward_function = {self.pump.reward_function}]')
    
    def __repr__(self):
        return f"<Well {self.index}>"
    
    def __lt__(self, other): # enables sorting wells by index
        return self.index < other.index
    
    def reward(self, override_checks:bool = False):
        """Deliver a Reward (using the paramters given at well initialization)"""
        self.pokes += 1
        self.reward_given = False
        if (self.state == well_states.active and time.time() - Well.last_reward > Well.reward_cooldown) or override_checks:
            Well.last_reward = time.time()
            self.rewards += 1
            self.reward_given = True
            self.pump.deliver_reward()
            self.state = well_states.reward_given
            ssi.disp(f"well {self.index} poked; reward given = 1")
        elif self.state == well_states.active:
            ssi.disp(f"""reward not delivered to active well {self.index} [port {self.pump_port}] 
                 because not enough time has elapsed since the last reward. 
                 Please check that reward wells are functioning well""".replace('\n', ''))
        elif self.state == well_states.reward_given:
            ssi.disp(f"well {self.index} must be reactivated before it can give a second reward")
        else:
            ssi.disp(f"well {self.index} poked; reward given = 0")
        return self.reward_given
    
    def activate(self, update_leds_immediately:bool = False):
        """An Active Well will Deliver Reward when Poked"""
        self.state = well_states.active
        self.reward_given = False
        if self.cued or update_leds_immediately: self.led.on(update_leds_immediately)
    
    def deactivate(self, update_leds_immediately:bool = False):
        """An Inactive Well will not Deliver Reward when Poked"""
        self.state = well_states.inactive
        if self.cued or update_leds_immediately: self.led.off(update_leds_immediately)

# A Callback Function which is Good for Reward-Well Mazes (like the ones we currently make/use)
def well_callback(line:str) -> int:
    """Parse a Line from StateScript, returning if a command 
    was matched (res&1==1), and if reward was given (res&2==2)
    """
    
    # Use the Base Callback Function
    match, t, command, value = ssi.parse_command(line)
    
    # Check the Result
    if match:
        # Will be set to True if a reward is given
        reward_given = False
        
        # Check if a Well was Specified
        well = None
        if value != None:
            # Get the Well
            well = wells[ssi.config.get_port_index(value)]
            
            # Give Reward (well.reward() checks if the well is active)
            if command == ssi.config.rewarded_command:
                reward_given = well.reward()
        
        # Call the Command Handler
        if well != None:
            ssi.config.command_handlers[command](t, well)
        else:
            ssi.config.command_handlers[command](t)
        
        # Display Stats
        ssi.print_stats()
        
        # Return if a Match was Found
        return 1 + 2*reward_given
    else:
        # Return that a Match was Not Found
        return 0

"""
Platforms for Building Mazes
"""

class FileDrivenMaze(ParameterFile):
    def __init__(self, filepath:str):
        # Initialize Attributes which Aren't Calculated From the Parameter File
        self.goal = None
        self.previous_goal = None
        self.reps_remaining = 0
        self.start_time = None
        self.timeout_grace_period = 0
        self.max_epochs_updated = False
        self.leds = []
        
        # Load the Parameter File (initializes several attributes)
        self.last_goal = -1
        self.alpha = 0.5
        self.beta = -0.5
        self.gamma = 0.8
        self.delta = 0.01
        super(FileDrivenMaze, self).__init__(filepath)
        
        # Initialize Attributes which Are Calculated from the Parameter File
        self.possible_goal_count = len(self.total_visits)
    
    def get_epoch_time(self) -> float:
        """calculates the time since the first home poke in minutes"""
        return (time.time() - self.start_time) / 60 if self.start_time != None else 0
    
    def timed_out(self) -> float:
        """checks if the epoch timed out (returns true if enough time has passed regardless of the number of trials completed)"""
        return self.get_epoch_time() > self.timeout + self.timeout_grace_period
    
    def get_outreps(self):
        if type(self.outreps) == int:
            return self.outreps
        elif type(self.outreps) == tuple and len(self.outreps) == 2:
            return randint(*self.outreps)
        elif type(self.outreps) == list and self.outreps:
            return choice(self.outreps)
        else:
            outreps = 1 if self.cued == 1 else 15
            ssi.disp(f"unrecognized value for outreps. defaulting to {outreps}")
            return outreps
    
    def _get_weighted_visits(self):
        all_pokes_this_epoch = np.array([w.pokes for w in self.outer_wells])
        rewarded_pokes_this_epoch = np.array([w.rewards for w in self.outer_wells])
        combined_pokes = self.alpha * rewarded_pokes_this_epoch + (1 - self.alpha) * (all_pokes_this_epoch - rewarded_pokes_this_epoch)
        return (1 - self.gamma) * combined_pokes + self.gamma * np.array(self.weighted_visits)
    
    def get_goal_probabilities(self, possible_goals:list):
        """Get the Selection Probability for Each Possible Goal"""
        # Check the Goal Selection Mode
        if self.goal_selection_mode <= 0:  # Choose a Goal(s) at Random (but not the previous goal)
            P = np.ones(len(possible_goals), int)
        elif self.goal_selection_mode == 1:
            rewarded_visits = np.array(self.rewarded_visits)
            total_visits = np.array(self.total_visits)
            visits = self.alpha * rewarded_visits + (1 - self.alpha)*(total_visits - rewarded_visits)
            return softmax(visits - np.min(visits) + 1, self.beta)
        elif self.goal_selection_mode == 2:
            weighted_visits = self._get_weighted_visits()
            return softmax(np.array(weighted_visits) - np.min(weighted_visits) + 1, self.beta)
        else:
            raise ValueError(f"unrecognized arm selection mode: {self.goal_selection_mode}")
        
        # Exclude the Current Goal
        excluded_goal = self.goal[0] if self.goals == 1 and self.goal else None
        if excluded_goal: P[excluded_goal.index] = 0
        
        # Apply the Lower Bound
        s = (np.sum(P) + len(P) * self.delta)
        P += self.delta / s
        
        # Normalize and Return the Probabilities
        return P / s # np.sum(P)
    
    def _disp_goal_block_info(self):
        # this ensures that this string is always idential (for ease of parsing later)
        ssi.disp(f"new goal block: [goal = {[w.index for w in self.goal]}, outreps = {self.reps_remaining}]")
    
    def select_goal(self, possible_goals:list):
        # Set the Previous Goal
        self.previous_goal = [] if self.goal == None else self.goal
        
        # Select the Number of Outreps
        self.reps_remaining = self.get_outreps()
        ssi.logger.add_line_break(char = '-')
        ssi.disp('new goal block', False)
        
        # Get the Possible Goals (if specified with a group)
        if type(possible_goals) in (str, int, tuple) and possible_goals in wells:
            possible_goals = wells[possible_goals]
        
        # Copy the Possible Goals (to not affect the original list)
        possible_goals = possible_goals.copy()
        
        # Get the Number of Goals to Select
        goals = max(self.goals, self.forageassist)
        
        # Check if All Options Need to be Selected
        if len(possible_goals) <= goals:
            self.goal = possible_goals
            self._disp_goal_block_info()
            return
        
        # Check the Arm Selection Mode
        goal = None
        if self.goals == 1 and self.last_goal > -1:
            self.goal = [possible_goals[self.last_goal - 1]]
        elif self.goal_selection_mode or self.goals != 1:
            # Get the Probability Distribution
            P = self.get_goal_probabilities(possible_goals)
            
            # Draw From the Distribution
            if self.goals == 1:
                # Draw Until Success
                goal = self.previous_goal
                while goal == self.previous_goal and len(possible_goals) > 1:
                    # Draw From the Distribution
                    goal = list(choice(possible_goals, 1, True, P))
                    
                    # Make the Distribution a Little Flatter
                    # (helps avoid cases where the distirbution is so skewed so badly that only 1 goal can be picked by np.choice)
                    P = 0.99 * P + 0.01
                    P /= np.sum(P)
            else:
                # Draw From the Distribution (it's ok for this to be the same as the previous goal)
                goal = list(choice(possible_goals, self.goals, False, P))
        else: # pre-scripted shuffled sequence (not really intended to be used for more than 1 goal)
            goal = [wells[self.goal_sequence.pop()]]
            
            # Generate a New Sequence if Necessary
            if not self.goal_sequence:
                # Initialize a New Goal-Sequence
                self.goal_sequence = [goal.index for goal in possible_goals]
                shuffle(self.goal_sequence)
                
                # Re-Shuffle the Sequence to Make Sure the Next Goal isn't the Current Goal
                if len(self.goal_sequence) > 1:
                    while wells[self.goal_sequence[-1]] == goal[0]:
                        shuffle(self.goal_sequence)
        
        # Make Sure the Goal is a List
        if type(goal) != list:
            if hasattr('__iter__'):
                goal = list(goal)
            else:
                goal = [goal]
        
        # Select Extra LEDs to be Lit
        if self.goals == 1: # and self.cues not in (0, 1):
            if self.cues != 0 and abs(self.cues) < len(possible_goals):
                self.leds.clear()
                self.leds.extend(goal)
                if abs(self.cues) > len(self.leds):
                    leds = [w for w in wells if w not in goal and w != self.home]
                    if self.previous_goal:
                        prev = self.previous_goal[0]
                        if prev in leds: 
                            leds.remove(prev)
                        if self.cues > 0:
                            self.leds.append(prev)
                    needed = abs(self.cues) - len(self.leds)
                    if needed > 0:
                        if needed < len(leds):
                            self.leds.extend(choice(leds, needed, False))
                        else:
                            self.leds.extend(leds)
            else:
                self.leds = possible_goals
        
        # Track the Selected Goal
        self.goal = goal
        if self.goals == 1 and self.last_goal > -1: self.last_goal = self.goal[0].index
        self.stats.this_goal = 0
        for g in goal: self.goal_counts[g.index - 1] += 1
        self._disp_goal_block_info()
        
        # Return the Selection
        return goal
    

"""
Other Functions
"""

# softmax distribution function with a "temperature" parameter
def softmax(X, beta):
    """returns the normalized vector e^(beta * X)"""
    X = np.exp(beta * np.array(X))
    return X / np.sum(X)