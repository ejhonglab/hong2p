
CREATE TABLE IF NOT EXISTS flies (
    /* TODO serial? work w/ pandas insert if just not specified? */
    fly SERIAL PRIMARY KEY,
    prep_date date,
    /* TODO also load fly_num column from google sheet and use that w/ prep_date
     * to generate fly PK? */
    /* TODO appropriate constraints on each of these? */
    indicator text,
    age integer,
    surgeon text,
    surgery_quality smallint,
    note text
);

CREATE TABLE IF NOT EXISTS odors (
    odor SERIAL PRIMARY KEY,
    /* TODO maybe use CAS or some other ID instead? */
    name text NOT NULL,
    log10_conc_vv real NOT NULL
);

CREATE TABLE IF NOT EXISTS mixtures (
    mixture SERIAL PRIMARY KEY,

    /* TODO TODO pandas sql interface work w/ arrays? r + matlab? */
    /* TODO check not zero length if using arrays here */
    /*
    name text[] NOT NULL,
    log10_conc_vv real[] NOT NULL,
    */

    /* TODO any way to let multiple odors be in one mixture without arrays? */

    pulse_length real NOT NULL,
    carrier_flow_slpm real NOT NULL,
    odor_flow_slpm real NOT NULL,
    volume_ml real NOT NULL
    /* TODO also pin / position? */
    /* TODO date / who made? / recipie? */
);

CREATE TABLE IF NOT EXISTS odors_in_mixtures (
    mixture integer REFERENCES mixtures (mixture),
    odor integer REFERENCES odors (odor),
    PRIMARY KEY(mixture, odor)
);

/* TODO table for stimulus info? */
/* TODO maybe stimulus code or version somewhere if nothing else? */

CREATE TABLE IF NOT EXISTS recordings (
    /*recording_num SERIAL PRIMARY KEY,*/
    /* TODO appropriate precision? */
    started_at timestamptz(0) PRIMARY KEY,
    /* TODO check thorsync and thorimage are unique (w/in day at least?)
       require longer path and just check unique across all?
    */
    /* TODO just use combination of paths as primary key */
    thorsync_path text NOT NULL,
    /* TODO check this is not null in the responses case? */
    thorimage_path text,
    /* TODO maybe require this if it's just going to be the pin/odor info? */
    stimulus_data_path text
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_description text NOT NULL,
    /* TODO precision? */
    analyzed_at timestamptz(0) NOT NULL,
    /* TODO actually any simpler making some kind of artificial key like this
     * for fk purposes? i didn't want to have to refer to two extra columns in
     * responses keys... */
    analysis_run SERIAL UNIQUE NOT NULL,
    /* TODO git version + unsaved changes? */
    PRIMARY KEY(analysis_description, analyzed_at)
);

CREATE TABLE IF NOT EXISTS responses (
    analysis integer REFERENCES analysis_runs (analysis_run) NOT NULL,

    fly integer REFERENCES flies (fly) NOT NULL,
    recording_from timestamptz(0) REFERENCES recordings (started_at) NOT NULL,
    /* rename? roi/footprint/component? */
    cell integer NOT NULL,

    mixture integer REFERENCES mixtures (mixture) NOT NULL,
    /* TODO positive / nonneg constraint. alt repr? */
    repeat_num integer NOT NULL,

    from_onset double precision NOT NULL,
    df_over_f real NOT NULL,

    PRIMARY KEY(analysis, fly, recording_from, cell, mixture, repeat_num, from_onset)
);

CREATE TABLE IF NOT EXISTS pid (
    mixture integer REFERENCES mixtures (mixture) NOT NULL,
    recording_from timestamptz(0) REFERENCES recordings (started_at) NOT NULL,
    /* TODO positive / nonneg constraint. alt repr? */
    repeat_num integer NOT NULL,

    from_onset double precision NOT NULL,
    pid_out real NOT NULL,

    PRIMARY KEY(mixture, recording_from, repeat_num, from_onset)
);
/* TODO link to table w/ PID / flow settings or just include for each row here?
 * */

/* TODO worth keeping a representation of traces across whole experiment?
   convert everything to odor responses?

   just include start frame / time for each chunk?
 */

/* TODO store footprints? how to represent? */

/* TODO maybe store info about the analysis that generated the given responses
   as well as the source data */
