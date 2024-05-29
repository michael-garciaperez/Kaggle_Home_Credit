from pathlib import Path
import gc
from glob import glob
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import pickle

import warnings

warnings.filterwarnings("ignore")


class Pipeline:
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
                df = df.with_columns(pl.col(col).dt.total_days())  # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (
                df[col].dtype == pl.String
            ):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


class Aggregator:
    # Please add or subtract features yourself, be aware that too many features will take up too much space.
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_last + expr_mean

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_last + expr_mean

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        # expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        return expr_max + expr_last  # +expr_count

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        return expr_max + expr_last

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        return expr_max + expr_last

    def get_exprs(df):
        exprs = (
            Aggregator.num_expr(df)
            + Aggregator.date_expr(df)
            + Aggregator.str_expr(df)
            + Aggregator.other_expr(df)
            + Aggregator.count_expr(df)
        )

        return exprs


def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        month_decision=pl.col("date_decision").dt.month(),
        weekday_decision=pl.col("date_decision").dt.weekday(),
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


def read_files(regex_path, depth=None):
    chunks = []

    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)

    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df


class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


def predict(data_path, pkl_path, ohe=None):
    """Predict homecredit data.

    Parameters
    ----------
    data_path : str
        Path to data directory containing parquet files.
    pkl_path : str
        Path to additional required data. Here, it's
        the weights column and model data, contained
        in a single json file. The column data is unpacked
        and the model data is loaded in xgboost.

    Returns
    -------
    df_submission : pd.DataFrame
        Prediction formated as required by the score function.
    """
    TEST_DIR = data_path 

    data_store = {
        "df_base": read_file(TEST_DIR / "test_base.parquet"),
        "depth_0": [
            read_file(TEST_DIR / "test_static_cb_0.parquet"),
            read_files(TEST_DIR / "test_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
            read_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
            read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
            read_file(TEST_DIR / "test_other_1.parquet", 1),
            read_file(TEST_DIR / "test_person_1.parquet", 1),
            read_file(TEST_DIR / "test_deposit_1.parquet", 1),
            read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
            read_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
            read_file(TEST_DIR / "test_applprev_2.parquet", 2),
            read_file(TEST_DIR / "test_person_2.parquet", 2),
        ],
    }

    df_test = feature_eng(**data_store)

    print("test data shape:\t", df_test.shape)

    del data_store
    gc.collect()

    Important_Columns = [
        "case_id",
        "WEEK_NUM",
        "month_decision",
        "weekday_decision",
        "credamount_770A",
        "applicationcnt_361L",
        "applications30d_658L",
        "applicationscnt_1086L",
        "applicationscnt_464L",
        "applicationscnt_867L",
        "clientscnt_1022L",
        "clientscnt_100L",
        "clientscnt_1071L",
        "clientscnt_1130L",
        "clientscnt_157L",
        "clientscnt_257L",
        "clientscnt_304L",
        "clientscnt_360L",
        "clientscnt_493L",
        "clientscnt_533L",
        "clientscnt_887L",
        "clientscnt_946L",
        "deferredmnthsnum_166L",
        "disbursedcredamount_1113A",
        "downpmt_116A",
        "homephncnt_628L",
        "isbidproduct_1095L",
        "mobilephncnt_593L",
        "numactivecreds_622L",
        "numactivecredschannel_414L",
        "numactiverelcontr_750L",
        "numcontrs3months_479L",
        "numnotactivated_1143L",
        "numpmtchanneldd_318L",
        "numrejects9m_859L",
        "sellerplacecnt_915L",
        "max_mainoccupationinc_384A",
        "max_birth_259D",
        "max_num_group1_9",
        "birthdate_574D",
        "dateofbirth_337D",
        "days180_256L",
        "days30_165L",
        "days360_512L",
        "firstquarter_103L",
        "fourthquarter_440L",
        "secondquarter_766L",
        "thirdquarter_1082L",
        "max_debtoutstand_525A",
        "max_debtoverdue_47A",
        "max_refreshdate_3813885D",
        "mean_refreshdate_3813885D",
        "pmtscount_423L",
        "pmtssum_45A",
        "responsedate_1012D",
        "responsedate_4527233D",
        "actualdpdtolerance_344P",
        "amtinstpaidbefduel24m_4187115A",
        "numinstlswithdpd5_4187116L",
        "annuitynextmonth_57A",
        "currdebt_22A",
        "currdebtcredtyperange_828A",
        "numinstls_657L",
        "totalsettled_863A",
        "mindbddpdlast24m_3658935P",
        "avgdbddpdlast3m_4187120P",
        "mindbdtollast24m_4525191P",
        "avgdpdtolclosure24_3658938P",
        "avginstallast24m_3658937A",
        "maxinstallast24m_3658928A",
        "avgmaxdpdlast9m_3716943P",
        "avgoutstandbalancel6m_4187114A",
        "avgpmtlast12m_4525200A",
        "cntincpaycont9m_3716944L",
        "cntpmts24_3658933L",
        "commnoinclast6m_3546845L",
        "maxdpdfrom6mto36m_3546853P",
        "datefirstoffer_1144D",
        "datelastunpaid_3546854D",
        "daysoverduetolerancedd_3976961L",
        "numinsttopaygr_769L",
        "dtlastpmtallstes_4499206D",
        "eir_270L",
        "firstclxcampaign_1125D",
        "firstdatedue_489D",
        "lastactivateddate_801D",
        "lastapplicationdate_877D",
        "mean_creationdate_885D",
        "max_num_group1",
        "last_num_group1",
        "max_num_group2_14",
        "last_num_group2_14",
        "lastapprcredamount_781A",
        "lastapprdate_640D",
        "lastdelinqdate_224D",
        "lastrejectcredamount_222A",
        "lastrejectdate_50D",
        "maininc_215A",
        "mastercontrelectronic_519L",
        "mastercontrexist_109L",
        "maxannuity_159A",
        "maxdebt4_972A",
        "maxdpdlast24m_143P",
        "maxdpdlast3m_392P",
        "maxdpdtolerance_374P",
        "maxdbddpdlast1m_3658939P",
        "maxdbddpdtollast12m_3658940P",
        "maxdbddpdtollast6m_4187119P",
        "maxdpdinstldate_3546855D",
        "maxdpdinstlnum_3546846P",
        "maxlnamtstart6m_4525199A",
        "maxoutstandbalancel12m_4187113A",
        "numinstpaidearly_338L",
        "numinstpaidearly5d_1087L",
        "numinstpaidlate1d_3546852L",
        "numincomingpmts_3546848L",
        "numinstlsallpaid_934L",
        "numinstlswithdpd10_728L",
        "numinstlswithoutdpd_562L",
        "numinstpaid_4499208L",
        "numinstpaidearly3d_3546850L",
        "numinstregularpaidest_4493210L",
        "numinstpaidearly5dest_4493211L",
        "sumoutstandtotalest_4493215A",
        "numinstpaidlastcontr_4325080L",
        "numinstregularpaid_973L",
        "pctinstlsallpaidearl3d_427L",
        "pctinstlsallpaidlate1d_3546856L",
        "pctinstlsallpaidlat10d_839L",
        "pctinstlsallpaidlate4d_3546849L",
        "pctinstlsallpaidlate6d_3546844L",
        "pmtnum_254L",
        "posfpd10lastmonth_333P",
        "posfpd30lastmonth_3976960P",
        "posfstqpd30lastmonth_3976962P",
        "price_1097A",
        "sumoutstandtotal_3546847A",
        "totaldebt_9A",
        "mean_actualdpd_943P",
        "max_annuity_853A",
        "mean_annuity_853A",
        "max_credacc_credlmt_575A",
        "max_credamount_590A",
        "max_downpmt_134A",
        "mean_credacc_credlmt_575A",
        "mean_credamount_590A",
        "mean_downpmt_134A",
        "max_currdebt_94A",
        "mean_currdebt_94A",
        "max_mainoccupationinc_437A",
        "mean_mainoccupationinc_437A",
        "mean_maxdpdtolerance_577P",
        "max_outstandingdebt_522A",
        "mean_outstandingdebt_522A",
        "last_actualdpd_943P",
        "last_annuity_853A",
        "last_credacc_credlmt_575A",
        "last_credamount_590A",
        "last_downpmt_134A",
        "last_currdebt_94A",
        "last_mainoccupationinc_437A",
        "last_maxdpdtolerance_577P",
        "last_outstandingdebt_522A",
        "max_approvaldate_319D",
        "mean_approvaldate_319D",
        "max_dateactivated_425D",
        "mean_dateactivated_425D",
        "max_dtlastpmt_581D",
        "mean_dtlastpmt_581D",
        "max_dtlastpmtallstes_3545839D",
        "mean_dtlastpmtallstes_3545839D",
        "max_employedfrom_700D",
        "max_firstnonzeroinstldate_307D",
        "mean_firstnonzeroinstldate_307D",
        "last_approvaldate_319D",
        "last_creationdate_885D",
        "last_dateactivated_425D",
        "last_dtlastpmtallstes_3545839D",
        "last_employedfrom_700D",
        "last_firstnonzeroinstldate_307D",
        "max_byoccupationinc_3656910L",
        "max_childnum_21L",
        "max_pmtnum_8L",
        "last_pmtnum_8L",
        "max_pmtamount_36A",
        "last_pmtamount_36A",
        "max_processingdate_168D",
        "last_processingdate_168D",
        "max_num_group1_5",
        "mean_credlmt_230A",
        "mean_credlmt_935A",
        "mean_pmts_dpd_1073P",
        "max_dpdmaxdatemonth_89T",
        "max_dpdmaxdateyear_596T",
        "max_pmts_dpd_303P",
        "mean_dpdmax_757P",
        "max_dpdmaxdatemonth_442T",
        "max_dpdmaxdateyear_896T",
        "mean_pmts_dpd_303P",
        "mean_instlamount_768A",
        "mean_monthlyinstlamount_332A",
        "max_monthlyinstlamount_674A",
        "mean_monthlyinstlamount_674A",
        "mean_outstandingamount_354A",
        "mean_outstandingamount_362A",
        "mean_overdueamount_31A",
        "mean_overdueamount_659A",
        "max_numberofoverdueinstls_725L",
        "mean_overdueamountmax2_14A",
        "mean_totaloutstanddebtvalue_39A",
        "mean_dateofcredend_289D",
        "mean_dateofcredstart_739D",
        "max_lastupdate_1112D",
        "mean_lastupdate_1112D",
        "max_numberofcontrsvalue_258L",
        "max_numberofoverdueinstlmax_1039L",
        "max_overdueamountmaxdatemonth_365T",
        "max_overdueamountmaxdateyear_2T",
        "mean_pmts_overdue_1140A",
        "max_pmts_month_158T",
        "max_pmts_year_1139T",
        "mean_overdueamountmax2_398A",
        "max_dateofcredend_353D",
        "max_dateofcredstart_181D",
        "mean_dateofcredend_353D",
        "max_numberofoverdueinstlmax_1151L",
        "mean_overdueamountmax_35A",
        "max_overdueamountmaxdatemonth_284T",
        "max_overdueamountmaxdateyear_994T",
        "mean_pmts_overdue_1152A",
        "max_residualamount_488A",
        "mean_residualamount_856A",
        "max_totalamount_6A",
        "mean_totalamount_6A",
        "mean_totalamount_996A",
        "mean_totaldebtoverduevalue_718A",
        "mean_totaloutstanddebtvalue_668A",
        "max_numberofcontrsvalue_358L",
        "max_dateofrealrepmt_138D",
        "mean_dateofrealrepmt_138D",
        "max_lastupdate_388D",
        "mean_lastupdate_388D",
        "max_numberofoverdueinstlmaxdat_148D",
        "mean_numberofoverdueinstlmaxdat_641D",
        "mean_overdueamountmax2date_1002D",
        "max_overdueamountmax2date_1142D",
        "last_refreshdate_3813885D",
        "max_nominalrate_281L",
        "max_nominalrate_498L",
        "max_numberofinstls_229L",
        "max_numberofinstls_320L",
        "max_numberofoutstandinstls_520L",
        "max_numberofoutstandinstls_59L",
        "max_numberofoverdueinstls_834L",
        "max_periodicityofpmts_1102L",
        "max_periodicityofpmts_837L",
        "last_num_group1_6",
        "last_mainoccupationinc_384A",
        "last_birth_259D",
        "max_empl_employedfrom_271D",
        "last_personindex_1023L",
        "last_persontype_1072L",
        "max_collater_valueofguarantee_1124L",
        "max_collater_valueofguarantee_876L",
        "max_pmts_month_706T",
        "max_pmts_year_507T",
        "last_pmts_month_158T",
        "last_pmts_year_1139T",
        "last_pmts_month_706T",
        "last_pmts_year_507T",
        "max_num_group1_13",
        "max_num_group2_13",
        "last_num_group2_13",
        "max_num_group1_15",
        "max_num_group2_15",
        "description_5085714M",
        "education_1103M",
        "education_88M",
        "maritalst_385M",
        "maritalst_893M",
        "requesttype_4525192L",
        "credtype_322L",
        "disbursementtype_67L",
        "inittransactioncode_186L",
        "lastapprcommoditycat_1041M",
        "lastcancelreason_561M",
        "lastrejectcommoditycat_161M",
        "lastrejectcommodtypec_5251769M",
        "lastrejectreason_759M",
        "lastrejectreasonclient_4145040M",
        "lastst_736L",
        "opencred_647L",
        "paytype1st_925L",
        "paytype_783L",
        "twobodfilling_608L",
        "max_cancelreason_3545846M",
        "max_education_1138M",
        "max_postype_4733339M",
        "max_rejectreason_755M",
        "max_rejectreasonclient_4145042M",
        "last_cancelreason_3545846M",
        "last_education_1138M",
        "last_postype_4733339M",
        "last_rejectreason_755M",
        "last_rejectreasonclient_4145042M",
        "max_credtype_587L",
        "max_familystate_726L",
        "max_inittransactioncode_279L",
        "max_isbidproduct_390L",
        "max_status_219L",
        "last_credtype_587L",
        "last_familystate_726L",
        "last_inittransactioncode_279L",
        "last_isbidproduct_390L",
        "last_status_219L",
        "max_classificationofcontr_13M",
        "max_classificationofcontr_400M",
        "max_contractst_545M",
        "max_contractst_964M",
        "max_description_351M",
        "max_financialinstitution_382M",
        "max_financialinstitution_591M",
        "max_purposeofcred_426M",
        "max_purposeofcred_874M",
        "max_subjectrole_182M",
        "max_subjectrole_93M",
        "last_classificationofcontr_13M",
        "last_classificationofcontr_400M",
        "last_contractst_545M",
        "last_contractst_964M",
        "last_description_351M",
        "last_financialinstitution_382M",
        "last_financialinstitution_591M",
        "last_purposeofcred_426M",
        "last_purposeofcred_874M",
        "last_subjectrole_182M",
        "last_subjectrole_93M",
        "max_education_927M",
        "max_empladdr_district_926M",
        "max_empladdr_zipcode_114M",
        "max_language1_981M",
        "last_education_927M",
        "last_empladdr_district_926M",
        "last_empladdr_zipcode_114M",
        "last_language1_981M",
        "max_contaddr_matchlist_1032L",
        "max_contaddr_smempladdr_334L",
        "max_empl_employedtotal_800L",
        "max_empl_industry_691L",
        "max_familystate_447L",
        "max_incometype_1044T",
        "max_relationshiptoclient_415T",
        "max_relationshiptoclient_642T",
        "max_remitter_829L",
        "max_role_1084L",
        "max_safeguarantyflag_411L",
        "max_sex_738L",
        "max_type_25L",
        "last_contaddr_matchlist_1032L",
        "last_contaddr_smempladdr_334L",
        "last_incometype_1044T",
        "last_relationshiptoclient_642T",
        "last_role_1084L",
        "last_safeguarantyflag_411L",
        "last_sex_738L",
        "last_type_25L",
        "max_collater_typofvalofguarant_298M",
        "max_collater_typofvalofguarant_407M",
        "max_collaterals_typeofguarante_359M",
        "max_collaterals_typeofguarante_669M",
        "max_subjectroles_name_541M",
        "max_subjectroles_name_838M",
        "last_collater_typofvalofguarant_298M",
        "last_collater_typofvalofguarant_407M",
        "last_collaterals_typeofguarante_359M",
        "last_collaterals_typeofguarante_669M",
        "last_subjectroles_name_541M",
        "last_subjectroles_name_838M",
        "max_cacccardblochreas_147M",
        "last_cacccardblochreas_147M",
        "max_conts_type_509L",
        "last_conts_type_509L",
        "max_conts_role_79M",
        "max_empls_economicalst_849M",
        "max_empls_employer_name_740M",
        "last_conts_role_79M",
        "last_empls_economicalst_849M",
        "last_empls_employer_name_740M",
    ]

    Categorical_Columns = [
        "description_5085714M",
        "education_1103M",
        "education_88M",
        "maritalst_385M",
        "maritalst_893M",
        "requesttype_4525192L",
        "credtype_322L",
        "disbursementtype_67L",
        "inittransactioncode_186L",
        "lastapprcommoditycat_1041M",
        "lastcancelreason_561M",
        "lastrejectcommoditycat_161M",
        "lastrejectcommodtypec_5251769M",
        "lastrejectreason_759M",
        "lastrejectreasonclient_4145040M",
        "lastst_736L",
        "opencred_647L",
        "paytype1st_925L",
        "paytype_783L",
        "twobodfilling_608L",
        "max_cancelreason_3545846M",
        "max_education_1138M",
        "max_postype_4733339M",
        "max_rejectreason_755M",
        "max_rejectreasonclient_4145042M",
        "last_cancelreason_3545846M",
        "last_education_1138M",
        "last_postype_4733339M",
        "last_rejectreason_755M",
        "last_rejectreasonclient_4145042M",
        "max_credtype_587L",
        "max_familystate_726L",
        "max_inittransactioncode_279L",
        "max_isbidproduct_390L",
        "max_status_219L",
        "last_credtype_587L",
        "last_familystate_726L",
        "last_inittransactioncode_279L",
        "last_isbidproduct_390L",
        "last_status_219L",
        "max_classificationofcontr_13M",
        "max_classificationofcontr_400M",
        "max_contractst_545M",
        "max_contractst_964M",
        "max_description_351M",
        "max_financialinstitution_382M",
        "max_financialinstitution_591M",
        "max_purposeofcred_426M",
        "max_purposeofcred_874M",
        "max_subjectrole_182M",
        "max_subjectrole_93M",
        "last_classificationofcontr_13M",
        "last_classificationofcontr_400M",
        "last_contractst_545M",
        "last_contractst_964M",
        "last_description_351M",
        "last_financialinstitution_382M",
        "last_financialinstitution_591M",
        "last_purposeofcred_426M",
        "last_purposeofcred_874M",
        "last_subjectrole_182M",
        "last_subjectrole_93M",
        "max_education_927M",
        "max_empladdr_district_926M",
        "max_empladdr_zipcode_114M",
        "max_language1_981M",
        "last_education_927M",
        "last_empladdr_district_926M",
        "last_empladdr_zipcode_114M",
        "last_language1_981M",
        "max_contaddr_matchlist_1032L",
        "max_contaddr_smempladdr_334L",
        "max_empl_employedtotal_800L",
        "max_empl_industry_691L",
        "max_familystate_447L",
        "max_incometype_1044T",
        "max_relationshiptoclient_415T",
        "max_relationshiptoclient_642T",
        "max_remitter_829L",
        "max_role_1084L",
        "max_safeguarantyflag_411L",
        "max_sex_738L",
        "max_type_25L",
        "last_contaddr_matchlist_1032L",
        "last_contaddr_smempladdr_334L",
        "last_incometype_1044T",
        "last_relationshiptoclient_642T",
        "last_role_1084L",
        "last_safeguarantyflag_411L",
        "last_sex_738L",
        "last_type_25L",
        "max_collater_typofvalofguarant_298M",
        "max_collater_typofvalofguarant_407M",
        "max_collaterals_typeofguarante_359M",
        "max_collaterals_typeofguarante_669M",
        "max_subjectroles_name_541M",
        "max_subjectroles_name_838M",
        "last_collater_typofvalofguarant_298M",
        "last_collater_typofvalofguarant_407M",
        "last_collaterals_typeofguarante_359M",
        "last_collaterals_typeofguarante_669M",
        "last_subjectroles_name_541M",
        "last_subjectroles_name_838M",
        "max_cacccardblochreas_147M",
        "last_cacccardblochreas_147M",
        "max_conts_type_509L",
        "last_conts_type_509L",
        "max_conts_role_79M",
        "max_empls_economicalst_849M",
        "max_empls_employer_name_740M",
        "last_conts_role_79M",
        "last_empls_economicalst_849M",
        "last_empls_employer_name_740M",
    ]

    df_test = df_test.select(Important_Columns)

    print("test data shape:\t", df_test.shape)

    df_test, _ = to_pandas(df_test, Categorical_Columns)
    df_test = reduce_mem_usage(df_test)

    gc.collect()

    with open(pkl_path, "rb") as file:
        model = pickle.load(file)

    df_test = df_test.drop(columns=["WEEK_NUM"])
    df_test = df_test.set_index("case_id")

    dtest = xgb.DMatrix(df_test, enable_categorical=True)
    y_pred = pd.DataFrame(model.predict(dtest), index=df_test.index)

    df_submission = y_pred.rename(columns={0: "score"})
    return df_submission
