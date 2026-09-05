# Synthetic Equity Data

This directory contains the bundled equity dataset used by `crypto_rebalance_strategy --asset-class equity`.

## Files

- `equity_model_panel_parts/part_1996_2010.parquet`: model-ready monthly panel for 1,000 synthetic stocks from 1996 through 2010.
- `equity_model_panel_parts/part_2011_2025.parquet`: model-ready monthly panel for 1,000 synthetic stocks from 2011 through 2025.
- `monthly_universe_1000_stocks_30_years.txt`: monthly universe membership exported by the generator.
- `date_level_factor_data_30_years.txt`: date-level synthetic factor data exported by the generator.
- `synthetic_factor_panel_manifest.txt`: source-generation manifest.

## Model Panel Schema

The Parquet parts contain the columns required by the equity strategy:

- keys: `date`, `asset_id`
- returns: `stock_return_1m`, `forward_return_1m`, `forward_return_3m`
- price/liquidity/cost fields: `close`, `price_adv_20`, `spread_bps`, `impact_coefficient`, `commission_bps`
- model/factor fields: `elastic_net_alpha`, `observed_factor_001` through `observed_factor_050`

The original raw text panel was approximately 720 MB. The bundled Parquet parts are the model-ready subset used by the strategy and are split into sub-100 MB files for normal Git hosting.
