
CREATE TABLE IF NOT EXISTS flies (
    /* TODO serial? work w/ pandas insert if just not specified? */
    /*fly smallserial UNIQUE NOT NULL, */

    prep_date date NOT NULL,
    /* The fly's order within the day. */
    fly_num smallint NOT NULL,

    /* TODO also load fly_num column from google sheet and use that w/ prep_date
     * to generate fly PK? */
    /* TODO appropriate constraints on each of these? */
    indicator text,
    days_old smallint,
    surgeon text,
    surgery_quality smallint,
    notes text,

    PRIMARY KEY(prep_date, fly_num)
);

CREATE TABLE IF NOT EXISTS odors (
    /* TODO maybe use CAS or some other ID instead? */
    name text NOT NULL,
    log10_conc_vv real NOT NULL,

    odor smallserial UNIQUE NOT NULL,

    PRIMARY KEY(name, log10_conc_vv)
);

/* TODO or just make fk based pk w/ all fields of each odor duplicated? */
CREATE TABLE IF NOT EXISTS mixtures (
    /* TODO or just use name + conc? */
    odor1 smallint REFERENCES odors (odor) NOT NULL,
    odor2 smallint REFERENCES odors (odor) NOT NULL,
    PRIMARY KEY(odor1, odor2)
);

/* TODO table for stimulus info? */
/* TODO maybe stimulus code or version somewhere if nothing else? */

CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_description text NOT NULL,
    /* TODO precision? i think at least seconds? otherwise whatever is
     * convenient to work from time.time() in python... */
    analyzed_at timestamp NOT NULL,
    /* TODO actually any simpler making some kind of artificial key like this
     * for fk purposes? i didn't want to have to refer to two extra columns in
     * responses keys... */
    analysis_run smallserial UNIQUE NOT NULL,

    /* TODO git version + unsaved changes? */
    /* TODO git remotes? */
    PRIMARY KEY(analysis_description)
);

CREATE TABLE IF NOT EXISTS recordings (
    /*recording_num smallserial PRIMARY KEY,*/
    /* TODO appropriate precision? */
    started_at timestamp PRIMARY KEY,
    /* TODO check thorsync and thorimage are unique (w/in day at least?)
       require longer path and just check unique across all?
    */

    /* TODO just use combination of paths as primary key */
    thorsync_path text NOT NULL,
    /* TODO check this is not null in the responses case? */
    /* This is nullable so that recordings can also be used for PID recordings.
     * */
    thorimage_path text,
    /* TODO maybe require this if it's just going to be the pin/odor info? */
    stimulus_data_path text

    /* TODO store framerate here? so analysis on movies can do stuff for fixed
     * amount of seconds w/o having to load xml? */
);

/* TODO table for frame indices used to define responses? store alongside
 * responses, particularly if from_onset and df_over_f can be make into array
 * types? make a separate presentations / blocks table again? */

CREATE TABLE IF NOT EXISTS presentations (
    prep_date timestamp NOT NULL,
    fly_num smallint NOT NULL,

    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,

    /* TODO maybe reference an odor pair here? */
    comparison smallint NOT NULL,

    odor1 smallint NOT NULL,
    odor2 smallint NOT NULL,

    repeat_num smallint NOT NULL,
    
    /* TODO make include other key frames too? or relegate to block info? */
    odor_onset_frame integer NOT NULL,
    odor_offset_frame integer NOT NULL,

    FOREIGN KEY(prep_date, fly_num) REFERENCES flies(prep_date, fly_num),
    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(prep_date, fly_num, recording_from,
                comparison, odor1, odor2, repeat_num)
);

CREATE TABLE IF NOT EXISTS responses (
    /* TODO matter whether fk is an ID vs all columns of composite pk in other
     * table, as far as space / speed performance? */
    /*
    presentation_id integer REFERENCES presentations (presentation_id),
    */
    /* TODO TODO move prep_date + fly_num to recording? and just merge w/
     * recording_from? */

    /* TODO maybe still store this, just don't use as part of primary key? */
    analysis smallint REFERENCES analysis_runs (analysis_run) NOT NULL,

    prep_date timestamp,
    fly_num smallint,

    /*fly integer REFERENCES flies (fly) NOT NULL, */
    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,

    comparison smallint NOT NULL,

    /*
    mixture integer REFERENCES mixtures (mixture) NOT NULL,
    */
    odor1 smallint,
    odor2 smallint,

    /* TODO positive / nonneg constraint. alt repr? */
    repeat_num smallint NOT NULL,

    /* rename? roi/footprint/component? */
    cell smallint NOT NULL,

    from_onset double precision NOT NULL,

    df_over_f real NOT NULL,
    raw_f real NOT NULL,

    FOREIGN KEY(prep_date, fly_num) REFERENCES flies(prep_date, fly_num),
    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(analysis, prep_date, fly_num, recording_from,
        comparison, odor1, odor2, repeat_num, cell, from_onset)
    /*
    PRIMARY KEY(analysis, fly, recording_from, cell, mixture, repeat_num,
        from_onset)
    */
    /*PRIMARY KEY(presentation_id, from_onset)*/
);

CREATE TABLE IF NOT EXISTS pid (
    /*
    mixture smallint REFERENCES mixtures (mixture) NOT NULL,
    */
    odor1 smallint,
    odor2 smallint,

    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,
    /* TODO positive / nonneg constraint. alt repr? */
    repeat_num smallint NOT NULL,

    from_onset double precision NOT NULL,
    pid_out real NOT NULL,

    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(odor1, odor2, recording_from, repeat_num, from_onset)
    /*
    PRIMARY KEY(mixture, recording_from, repeat_num, from_onset)
    */
);
/* TODO link to table w/ PID / flow settings or just include for each row here?
 * */

/* TODO worth keeping a representation of traces across whole experiment?
   convert everything to odor responses?

   just include start frame / time for each chunk?
 */

/* TODO store footprints? how to represent? */
