-- `mozfun.hist.merge`(ARRAY_AGG(`mozfun.hist.extract`(h)))
-- hmerge(ARRAY_AGG(hist(h)))
/*
unq_tabs: {"bucket_count":50,"histogram_type":0,"sum":2,"range":[1,100],"values":{"0":0,"1":2,"2":0}}
*/

CREATE TEMP FUNCTION null_hist_str100() AS ('{"bucket_count":50,"histogram_type":0,"sum":0,"range\":[1,100]}');
CREATE TEMP FUNCTION null_hist_str10000() AS ('{"bucket_count":50,"histogram_type":0,"sum":0,"range":[1,10000]}');
CREATE TEMP FUNCTION slug() AS ('bug-1622934-pref-webrender-continued-v2-nightly-only-nightly-76-80');


CREATE TEMP FUNCTION hist100(h ANY TYPE) AS (`mozfun.hist.extract`(coalesce(h, null_hist_str100())));
CREATE TEMP FUNCTION hist1e4(h ANY TYPE) AS (`mozfun.hist.extract`(coalesce(h, null_hist_str10000())));

CREATE TEMP FUNCTION hmerge(h ANY TYPE) AS (`mozfun.hist.merge`(h));
CREATE TEMP FUNCTION get_key(hist ANY TYPE, k string) AS (`mozfun.map.get_key`(hist, k));
CREATE TEMP FUNCTION norm(h ANY TYPE) AS (
  `mozfun.hist.normalize`(h)
);

CREATE TEMP FUNCTION major_vers(st string) AS (
  -- '10.0' => 10
  cast(regexp_extract(st, '(\\d+)\\.?') as int64)
);



CREATE TEMP TABLE hists_per_cid
    as (
        with base as (
        select
          m.client_id as cid,
        --   submission_timestamp as ts,
          coalesce(m.environment.system.gfx.features.wr_qualified.status, '') = 'available' as wr_av,
          get_key(m.environment.experiments, slug()).branch is null as no_wr_exp,
          get_key(m.environment.experiments, slug()).branch as br,
          sample_id,
          date(m.submission_timestamp) as date,

          m.payload.histograms.fx_number_of_unique_site_origins_all_tabs as unq_tabs,
          m.payload.histograms.fx_number_of_unique_site_origins_per_document as unq_sites_per_doc,
          m.payload.histograms.cycle_collector as cycle_collector,
          m.payload.histograms.cycle_collector_max_pause as cycle_collector_max_pause,
          m.payload.histograms.cycle_collector_slice_during_idle as cycle_collector_slice_during_idle,
          -- cycle_collector_slice_during_idle
          m.payload.histograms.gc_max_pause_ms_2 as gc_max_pause_ms_2,
          m.payload.histograms.gc_ms as gc_ms,
          m.payload.histograms.gc_slice_during_idle as gc_slice_during_idle,

        from `moz-fx-data-shared-prod.telemetry.main` m
        where
          date(m.submission_timestamp) between '{sub_date_start}' and '{sub_date_end}'
          and m.normalized_channel = 'nightly'
          and m.normalized_app_name = 'Firefox'
          and sample_id between 1 and 1
        )

        select
          cid,
          date,
          br,
          count(distinct(cid)) as n_cid,

          -- 100
          norm(hmerge(array_agg(hist100(unq_tabs)))) as unq_tabs,
          norm(hmerge(array_agg(hist100(unq_sites_per_doc)))) as unq_sites_per_doc,
          norm(hmerge(array_agg(hist100(cycle_collector_slice_during_idle)))) as cycle_collector_slice_during_idle,
          norm(hmerge(array_agg(hist100(gc_slice_during_idle)))) as gc_slice_during_idle,

        --   -- 10,000
          norm(hmerge(array_agg(hist1e4(cycle_collector)))) as cycle_collector,
          norm(hmerge(array_agg(hist1e4(cycle_collector_max_pause)))) as cycle_collector_max_pause,
          norm(hmerge(array_agg(hist1e4(gc_max_pause_ms_2)))) as gc_max_pause_ms_2,
          norm(hmerge(array_agg(hist1e4(gc_ms)))) as gc_ms,

        from base
        where br is not null
        group by 1, 2, 3
);
    


with day_hist as (
select
  date,
  br,
  sum(n_cid) as n_cid,
  hmerge(array_agg(unq_tabs)).values as unq_tabs,
  hmerge(array_agg(unq_sites_per_doc)).values as unq_sites_per_doc,
  hmerge(array_agg(cycle_collector_slice_during_idle)).values as cycle_collector_slice_during_idle,
  hmerge(array_agg(gc_slice_during_idle)).values as gc_slice_during_idle,
  
  hmerge(array_agg(cycle_collector)).values as cycle_collector,
  hmerge(array_agg(cycle_collector_max_pause)).values as cycle_collector_max_pause,
  hmerge(array_agg(gc_max_pause_ms_2)).values as gc_max_pause_ms_2,
  hmerge(array_agg(gc_ms)).values as gc_ms,
from hists_per_cid
group by 1, 2
)

select *
from day_hist

