CREATE TEMP FUNCTION build_date(app_build_id string) AS (
  parse_date('%Y%m%d', substr(app_build_id, 1, 8))
);
CREATE TEMP FUNCTION slug() AS ('bug-1622934-pref-webrender-continued-v2-nightly-only-nightly-76-80');
CREATE TEMP FUNCTION get_key(hist ANY TYPE, k string) AS (`mozfun.map.get_key`(hist, k));



with base as (
select
  c.client_id,
  get_key(c.environment.experiments, slug()).branch as branch,
  IF(
    coalesce(payload.process_type, 'main') = 'main',
    1, 0
  ) AS main_crash,
  IF(
    REGEXP_CONTAINS(payload.process_type, 'content')
    AND NOT REGEXP_CONTAINS(COALESCE(payload.metadata.ipc_channel_error, ''), 'ShutDownKill'),
    1,
    0
  ) AS content_crash,
  IF(payload.metadata.startup_crash = '1', 1, 0) AS startup_crash,
  IF(
    REGEXP_CONTAINS(payload.process_type, 'content')
    AND REGEXP_CONTAINS(payload.metadata.ipc_channel_error, 'ShutDownKill'),
    1,
    0
  ) AS content_shutdown_crash,
  build_date(c.environment.build.build_id) as build_date,
from `telemetry.crash` c
where
  date(c.submission_timestamp) = '2020-03-01'
--   and sample_id = 1
  and c.application.name = 'Firefox'
  and c.application.channel = 'nightly'
--   and get_key(c.environment.experiments, slug()).branch is not null
)


, gb as (
select
  build_date,
  branch,
  sum(main_crash) as main_crash,
  sum(content_crash) as content_crash,
  sum(startup_crash) as startup_crash,
  sum(content_shutdown_crash) as content_shutdown_crash,
  
  /*
    the following counts # of unique crashing clients who have a main crash.
    You can change this to content_crash if you want. I'm not sure which they'll
    be interested in.
    
    This query groups by branch and build_date. branch is null here for some reason.
    You'll want to change it to the right slug for fission.
    I parsed build dates, since Nika said that was of greater interest than submission
    or crash date.
  */
  count(distinct(if(main_crash = 1, client_id, null))) as n_unique_clients_main_crash,
from base
group by 1, 2
)

select
  *,
from gb
