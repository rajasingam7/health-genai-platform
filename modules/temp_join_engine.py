def create_temp_join(df_profile_features, df_activity_agg):
    return df_profile_features.merge(
        df_activity_agg,
        on="Patient_Number",
        how="inner"
    )
