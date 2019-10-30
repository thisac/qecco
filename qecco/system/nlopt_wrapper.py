import nlopt
import numpy as np
import time
import numbers

returnCodeMessages = {
    1: "Generic success",
    2: "Reached stopval",
    3: "Reached ftol",
    4: "Reached xtol",
    5: "Reached maxeval",
    6: "Reached maxtime",
    -1: "Generic failure",
    -2: "Inavalid arguments",
    -3: "Out of memory",
    -4: "Roundoff limited",
    -5: "Forced stop",
    }


def get_initial_step(guess, lowerBounds, upperBounds):
    N = len(guess)
    steps = np.zeros((N, ))
    for i in range(N):
        step = (upperBounds[i] - lowerBounds[i]) * 0.25

        if (upperBounds[i] - guess[i]) < step:
            step = (upperBounds[i] - guess[i]) * 0.75

        if (guess[i] - lowerBounds[i]) < step:
            step = (guess[i] - lowerBounds[i]) * 0.75

        if np.isinf(step) or (step == 0.0):
            step = 1.0

        steps[i] = step
    return steps


def nlopt_optimize(error_fn, opt_algoritm, lowerBounds, upperBounds, **kwargs):
    """Optimize (default: minimize) error_fn using the input algorithm

Returns: Dictionary with results and other useful information (see below)

Required inputs:
  error_fn: Function that takes array of same dimension as bounds and
            returns a real number. This function will be minimized by
            default. Input will be a numpy array even if bounds are provided
            as a different type.
  lowerBounds: List, array or (for 1D problems) number setting the lower
               boundary for input to error_fn.
  upperBounds: List, array or (for 1D problems) number setting the upper
               boundary for input to error_fn.

Optional keyword inputs:
 nlopt stopping conditions:
  ftolRel: Relative function tolerance [default: None]
  ftolAbs: Absolute function tolerance [default: None]
  xtolRel: Relative tolerance of x [default: None]
  xtolAbs: Absolute tolerance of x [default: None]
  initialStep: Size of initial step for BOBYQA to take.
               Can be scalar or vector [default: None]
  maxEval: Max number of error function evaluations [default: None]
  maxTime: Max time to spend running the optimization [default: None]

 Wrapper-specific options:
  trackErrors: Whether to keep track of best error so far [default: True]
               Note that in case of error this saves last best evaluation
  guess: Optionally, provide a guess. [default: Chosen uniformly at random]
  minimize: If true, minimizes return from error_fn [default: True]
  printEvery: If not None, prints best error so far every n evaluations [default: None]

Note: kwargs will print a warning to stdout if an unrecognized keyword
is received but will keep going.

Results dictionary keys:
  'N': Dimensionality of x
  'lowerBounds': Provided lower bounds, converted to numpy array if necessary
  'upperBounds': Provided upper bounds, converted to numpy array if necessary

  'guess': The provided or randomly selected guess used as starting point
  'bestX': Function input that produced the lowest error
  'bestError': error_fn(bestX)
  'returnCode': Return value from the nlopt call
  'returnCodeMessage': Return value converted to its actual meaning

  'errorArray': 2D array containing the evaluation number at which a new
                best error was found, and the corresponding new best error.
                Last will be the final total number of evaluations, so
                the same best error may be repeated between the second
                to last and the last entry
  'numEvaluations': Final number of total evaluations

  'startUnixTime': Starting time of the optimization, unix time format
  'startHumanDate': Starting day of the optimization, human readable
  'startHumanTime': Starting time of the optimization, human readable

  'kwargs': The dictionary of keywords supplied when the function was called
"""
    # Get time function was called for recording purposes
    startTime = time.localtime()
    startUnixTime = time.mktime(startTime)
    startHumanDate = "{:04d}/{:02d}/{:02d}".format(
        startTime.tm_year, startTime.tm_mon, startTime.tm_mday)
    startHumanTime = "{:02d}:{:02d}:{:02d}".format(
        startTime.tm_hour, startTime.tm_min, startTime.tm_sec)

    # Ensure bounds are numpy arrays
    if isinstance(lowerBounds, numbers.Number):
        lowerBounds = np.array([lowerBounds])
    else:
        lowerBounds = np.array(lowerBounds)

    if isinstance(upperBounds, numbers.Number):
        upperBounds = np.array([upperBounds])
    else:
        upperBounds = np.array(upperBounds)

    # Get dimensionality of the problem
    N = lowerBounds.size
    assert upperBounds.size == N

    # Build an opt object and set bounds
    opt = nlopt.opt(opt_algoritm, N)
    opt.set_lower_bounds(lowerBounds)
    opt.set_upper_bounds(upperBounds)

    # Set options using kwargs
    # First, set some defaults
    trackErrors = True
    guess = None
    minimize = True

    # Next, check for known keys
    if 'vectorStorage' in kwargs:
        opt.set_vector_storage(kwargs['vectorStorage'])

    if 'stopval' in kwargs:
        opt.set_stopval(kwargs['stopval'])

    if 'ftolRel' in kwargs:
        opt.set_ftol_rel(kwargs['ftolRel'])

    if 'ftolAbs' in kwargs:
        opt.set_ftol_abs(kwargs['ftolAbs'])

    if 'xtolRel' in kwargs:
        opt.set_xtol_rel(kwargs['xtolRel'])

    if 'xtolAbs' in kwargs:
        opt.set_xtol_abs(kwargs['xtolAbs'])

    if 'initialStep' in kwargs:
        initialStep = np.array(kwargs['initialStep'])
        if initialStep.size == 1:
            initialStep = initialStep * np.ones((N, ))
        elif initialStep.size != N:
            raise IOError("Initial step vector is of wrong dimension")
        opt.set_initial_step(initialStep)

    if 'maxEval' in kwargs:
        opt.set_maxeval(int(kwargs['maxEval']))

    if 'maxTime' in kwargs:
        opt.set_maxtime(kwargs['maxTime'])

    if 'trackErrors' in kwargs:
        trackErrors = kwargs['trackErrors']

    if 'minimize' in kwargs:
        minimize = kwargs['minimize']

    if 'printEvery' in kwargs:
        printEvery = kwargs['printEvery']
    else:
        printEvery = None

    if 'guess' in kwargs:
        guess = kwargs['guess']
    else:
        # If a guess isn't provided, generate one as a uniform distribution
        # between the bounds by scaling and shifting a uniform dist
        # between 0 and 1 (the default of np.random.uniform)
        magnitude = upperBounds - lowerBounds
        mean = (upperBounds + lowerBounds) / 2.0
        guess = magnitude * np.random.uniform(low=-0.5, high=0.5, size=N) + mean

    # Warn about unrecognized keys
    knownKeys = ['stopval', 'vectorStorage', 'ftolRel', 'ftolAbs', 'xtolRel', 'xtolAbs', 'maxEval',
                 'maxTime', 'initialStep', 'trackErrors', 'minimize', 'printEvery',
                 'guess']
    for key in kwargs.keys():
        if key not in knownKeys:
            print("Warning: Got unrecognized keyword argument '{}'".format(key))

    # If we want to track the errors of this optimization, generate a
    # closure wrapping error_fn
    if trackErrors:
        global errors
        global minError
        global numEvaluations
        global lastBestX

        errors = []
        if minimize:
            minError = np.inf
        else:
            minError = -np.inf
        numEvaluations = 0
        lastBestX = None

        def wrapped_error_fn(x, grad):
            global numEvaluations
            global minError
            global numEvaluations
            global lastBestX
            error = error_fn(x, grad)
            numEvaluations += 1
            if (minimize and (error < minError)) or (not(minimize) and (error > minError)):
                errors.append([numEvaluations, error])
                minError = error
                lastBestX = x
            if printEvery is not None:
                if numEvaluations % printEvery == 0:
                    print("Error at {} evaluations: {:.2e} (best so far is {:.2e})".format(numEvaluations, error, minError))
            # if minError < 1e-12:
            #     print("Success: error is very, very small")
            #     raise nlopt.ForcedStop
            return error

    else:
        errors = None
        minError = None
        numEvaluations = None

        # If we're not tracking the results, check if error_fn already
        # obeys the nlopt calling convention of f(x, grad=[])
        try:
            startError = error_fn(guess, grad=[])
            # If it does, alias wrapped_error_fn to error_fn directly
            wrapped_error_fn = error_fn
        except TypeError:
            # If it doesn't, wrap error_fn in a closure that does obey
            # the calling convention
            def wrapped_error_fn(x, grad=[]):
                return error_fn(x)

    if minimize:
        opt.set_min_objective(wrapped_error_fn)
    else:
        opt.set_max_objective(wrapped_error_fn)

    # Finally, run the optimization
    caughtError = None
    bestX = None

    # NOTE: Set next line under try and remove pass
    # bestX = opt.optimize(guess)
    try:
        bestX = opt.optimize(guess)
    # Catch possible errors and save the nlopt website's description of them
    except RuntimeError:
        caughtError = "RuntimeError: Generic Failure"
    except ValueError:
        caughtError = "ValueError: Invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera)"
    except MemoryError:
        caughtError = "MemoryError: Ran out of memory (a memory allocation failed)"
    except nlopt.RoundoffLimited:
        caughtError = "nlopt.RoundoffLimited: Halted because roundoff errors halted progress"
    except nlopt.ForcedStop:
        caughtError = "nlopt.ForcedStop: Halted because of a forced termination: \
        The user called opt.force_stop() from the user's objective function or threw \
        an nlopt.ForcedStop exception."
    if caughtError is not None:
        bestX = lastBestX

    bestError = opt.last_optimum_value()
    returnCode = opt.last_optimize_result()
    returnCodeMessage = returnCodeMessages[returnCode]

    # Append one final result to the errors array with the final evaluation number
    if trackErrors:
        errors.append((numEvaluations, bestError))

    runTime = time.time() - startUnixTime

    # Put together a dictionary with the relevant data from this optimization
    results = {
        'N': N,
        'lowerBounds': lowerBounds,
        'upperBounds': upperBounds,
        'guess': guess,

        'bestX': bestX,
        'bestError': bestError,
        'returnCode': returnCode,
        'returnCodeMessage': returnCodeMessage,
        'caughtError': caughtError,

        'errorArray': np.array(errors),
        'numEvaluations': numEvaluations,

        'startUnixTime': startUnixTime,
        'runTime': runTime,
        'startHumanDate': startHumanDate,
        'startHumanTime': startHumanTime,
        'kwargs': kwargs,
        }
    return results
