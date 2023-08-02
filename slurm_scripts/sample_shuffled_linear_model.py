''' This script is intended to shift the neural data by a random amount, then select the central 5000 timepoints (to avoid edge effects between beginning and end of the data) and calculate the fit of a linear model to it. This will sample a distribution of weights and fit qualities that can be used to estimate significance of the paramters found for the non-shifted data'''
import matplotlib
import joblib
from joblib import Parallel, delayed
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import median_absolute_error
from sklearn.metrics import PredictionErrorDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
import scipy as sp
import numpy as np
import pandas as pd

class EvaluateLinear():

  def __init__(self, model, X_train, y_train, X_test, y_test):
    self.mae_train = median_absolute_error(y_train, model.predict(X_train))
    self.y_pred = model.predict(X_test)
    self.y_test = y_test
    self.residuals = self.y_test - self.y_pred
    self.mae_test = median_absolute_error(self.y_test, self.y_pred)

  def plot_residuals_over_time(self, ax=None):
    ''''''

    if ax is None:
        _, ax = ax.subplots(figsize=(14, 8))
    ax.plot(self.residuals)
    ax.set_title("Residuals over Time")
    return ax

  def plot_histogram_of_residuals(self, ax=None):
    ''''''
    if ax is None:
        _, ax = ax.subplots(figsize=(8, 6))
    ax.hist(self.residuals, bins=30)
    ax.set_title("Histogram of Residuals")
    return ax

  def plot_autocorrelation_of_residuals(self, ax=None):
    ''''''
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    ax = autocorrelation_plot(self.residuals, ax=ax)
    ax.set_title("Autocorrelation of Residuals")
    return ax

  def plot_paired_pred_test(self, ax=None):
    ''''''

    scores = {
        "MedAE on training set": f"{self.mae_train:.2f}",
        "MedAE on testing set": f"{self.mae_test:.2f}",
    }
    if ax is None:
        _, ax = ax.subplots(figsize=(5, 5))
    display = PredictionErrorDisplay.from_predictions(
        self.y_test, self.y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
    )
    ax.set_title("Ridge model, small regularization")
    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")
    return ax

  def plot_predicted_vs_test(self, ax=None):
    ''''''
    if ax is None:

      _, ax = ax.subplots(figsize=(14, 8))
    ax.plot(self.y_pred[:100], label='yhat')
    ax.plot(self.y_test[:100], label='y_test')
    ax.legend()
    ax.set_title('Model predicition of held-out test data')
    return ax

  def compose_plots(self, export_path: str) -> None:
    '''
    Combine all plots into a single figure, put out a pdf.


    '''
    with PdfPages(export_path) as pp:
        fig = plt.figure(figsize=(11.69291*2, 8.267717*2)) # A4 values in inches *2, horizontal
        gs = gridspec.GridSpec(30, 18, left=0.05, right=.95, top=.92, bottom=.05, wspace=0.00, hspace=0.00)

        ax_pred = fig.add_subplot(gs[0:10, :])
        ax_restime = fig.add_subplot(gs[11:20, :])
        ax_paired  = fig.add_subplot(gs[21:27, :5])
        ax_hist = fig.add_subplot(gs[21:27, 6:11])
        ax_corr = fig.add_subplot(gs[21:27, 12:18])

        self.plot_predicted_vs_test(ax_pred)
        self.plot_residuals_over_time(ax_restime)
        self.plot_paired_pred_test(ax_paired)
        self.plot_histogram_of_residuals(ax_hist)
        self.plot_autocorrelation_of_residuals(ax_corr)

        # Save the current page to the PDF file
        pp.savefig()
        plt.close()

def find_optimal_alpha(X_train, y_train, alpha=None):
  preprocessor = make_column_transformer(
    (StandardScaler(), list(range(X_train.shape[1]))), # does the same as zscore + normalize to vector length
    verbose_feature_names_out=False,  # avoid to prepend the preprocessor names
  )

  if alpha is not None:
      best_log_alpha = alpha
      tscv = TimeSeriesSplit(n_splits=5)
  else:
      alphas_iter_1 = np.logspace(-10, 1, 50)
      tscv = TimeSeriesSplit(n_splits=5)

      model = make_pipeline(
          preprocessor,
          RidgeCV(alphas=alphas_iter_1, cv=tscv),
      )

      model.fit(X_train, y_train)
      best_log_alpha = model.named_steps['ridgecv'].alpha_

  pow = int(np.log10(best_log_alpha))

  # Re-fit more precisely
  start = best_log_alpha - 10**pow
  stop = best_log_alpha + 10**pow
  alphas_iter_2 = np.linspace(start, stop, 3)
  alphas_iter_2 = alphas_iter_2[alphas_iter_2 > 0]

  model = make_pipeline(
      preprocessor,
      RidgeCV(alphas=alphas_iter_2, cv=tscv),
  )

  model.fit(X_train, y_train)
  return model

def circular_shift_model(X: np.ndarray, y: np.ndarray, n_shifts: int = 10, alpha=None) -> np.ndarray:
    '''
    Estimate true weight distribution by circularly shifting the stimulus and fitting the model to each shift.

    Parameters

    '''
    # Initialize array to hold the weight distributions
    weight_distributions = np.zeros((X.shape[1], n_shifts))
    alphas = np.zeros(n_shifts)
    maes = np.zeros(n_shifts)
    y = np.squeeze(y)[1009:-1009]
    print(f"y shape : {y.shape}")

    for i in range(n_shifts):

        # Only take the middle 5000 frames, to avoid edge effects
        shift = np.random.randint(0, X.shape[1]-5000)

        print(f"X shape : {X.shape}")
        X_iter = np.roll(X, shift, axis=1)
        X_iter = X_iter[1009:-1009, :]
        print(f"X_iter shape : {X_iter.shape}")

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_iter, y, test_size=0.5, shuffle=False)

        # Fit the model
        model = find_optimal_alpha(X_train, y_train, alpha)


        mae_train = median_absolute_error(y_train, model.predict(X_train))
        y_pred = model.predict(X_test)
        weight_distributions[:, i] = model.named_steps['ridgecv'].coef_
        alphas[i] = model.named_steps['ridgecv'].alpha_
        maes[i] = median_absolute_error(y_test, y_pred)

    return weight_distributions, alphas, maes



def process_behavior(behavior, name):
  save_path = "/scratch/wiessall/simple"
  y = behavior

  X_train, X_test, y_train, y_test = train_test_split(X[1009:-1009, :], y[1009:-1009], test_size=0.5, shuffle=False)
  model = find_optimal_alpha(X_train, y_train)
  alpha =  model.named_steps['ridgecv'].alpha_
  weight_distributions, alphas, maes = circular_shift_model(X, y, n_shifts=n_shifts, alpha=alpha)# Save the 2D array to a .npy file
  filename = f"{name}_weights.npy"
  np.save(filename, weight_distributions)

  # create a list of column names
  alpha_cols = [f'alpha_{i}' for i in range(n_shifts)]
  mae_cols = [f'mae_{i}' for i in range(n_shifts)]

  # create separate columns for each alpha and mae
  data = {col: [alpha]*X.shape[1] for col, alpha in zip(alpha_cols, alphas)}
  data.update({col: [mae]*X.shape[1] for col, mae in zip(mae_cols, maes)})

  df = pd.DataFrame(data, index=list(range(X.shape[1])))

  # add the remaining columns
  df['weight_distributions'] = [filename]*X.shape[1]
  df['model_weights'] = model.named_steps['ridgecv'].coef_
  df['model_alphas'] =  [alpha]*X.shape[1]
  df['model_maes'] = [median_absolute_error(y_train, model.predict(X_train))]*X.shape[1]

  # Save the DataFrame to a .parquet file without compression
  df.to_parquet(f"{save_path}/{name}_data.parquet", compression=None)

  joblib.dump(model, f"{save_path}/{name}_model.pkl")

  evaluator = EvaluateLinear(model, X_train, y_train, X_test, y_test)
  evaluator.compose_plots(export_path=f"{save_path}/{name}_evaluate.pdf")


save_path = "/scratch/wiessall/simple"
#Take only central part of the data
dat = np.load(f'{save_path}/stringer_spontaneous.npy', allow_pickle=True).item()
pupil = dat['pupilArea']
eye_pos = dat['pupilCOM']
locomotion = dat['run']
face_svd = dat['beh_svd_time']

# Obtain the euclidian distance between consecutive eye COM positions
diff = np.diff(eye_pos, axis=0)
sq_diff = diff**2
sq_distance = np.sum(sq_diff, axis=1)
distance = np.sqrt(sq_distance)
eye_speed = distance
eye_speed = np.pad(eye_speed, (0,1), mode='mean')



def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
nans, x= nan_helper(eye_speed)
eye_speed[nans]= np.interp(x(nans), x(~nans), eye_speed[~nans])

face = [row for row in face_svd.T]
behaviors = [pupil, eye_speed, locomotion]
behaviors.extend(face)
face_name = [f"face_{i}" for i in range(1000)]
behavior_names = ["pupil", "eye_position", "locomotion"]
behavior_names.extend(face_name)
####
X = dat['sresp'].T
n_shifts = 50
Parallel(n_jobs=1)(delayed(process_behavior)(behavior, name) for behavior, name in zip(behaviors[:], behavior_names[:]))
