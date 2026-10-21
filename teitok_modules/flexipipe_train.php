<?php
/**
 * Training models on the corpus — its own page (act=training): a training that is running (followed
 * live), the corpus's models (`flexipipe info corpus-models`: when, which backend, held-out scores,
 * which one is in use), what the corpus has to learn from and which backends can train on it
 * (`flexipipe info training`), and the form. Training runs as a background job (flexipipe_jobs.php).
 * Plan: dev/TEITOK-TRAINING.md.
 *
 * - Every training writes a new folder, Resources/flexipipe-models/<YYYYMMDD-HHMMSS>/ (with
 *   flexipipe-model.json: backend, model, scores): a model is never overwritten.
 * - Fresh splits every time (tmp/flexipipe-splits/<same name>/): what was corrected since counts.
 * - The model in use (<flexipipe><training use>; none set: the newest) gives the steps the corpus's
 *   main model does not do (pronunciation); "Tag with it" makes it the corpus's main model. Starting
 *   a retraining pins the model in use; "Use this model" switches, also where the corpus names the
 *   previous one as its main model.
 * Included by flexipipe.php; the set-up page links here.
 */

const FLEXIPIPE_MODELS_DIR = 'Resources/flexipipe-models';
const FLEXIPIPE_SPLITS_DIR = 'tmp/flexipipe-splits';

/** The corpus's models, newest first (not cached: they change with every training). */
function flexipipe_corpus_models( $fp_cmd ) {
	$data = $fp_cmd->json( array( 'info', 'corpus-models', '--corpus', '.', '--output-format', 'json' ), false );
	return $data['models'] ?? array();
}

/** What the corpus has to learn from, and which backends can train on it. */
function flexipipe_training_options( $fp_cmd ) {
	// reads a sample of the documents: kept 10 minutes
	return $fp_cmd->json( array( 'info', 'training', '--corpus', '.', '--output-format', 'json' ), 600 );
}

function flexipipe_rmtree( $dir ) {
	if ( !is_dir( $dir ) || is_link( $dir ) ) return @unlink( $dir );
	foreach ( (array)scandir( $dir ) as $f ) {
		if ( $f === '.' || $f === '..' ) continue;
		flexipipe_rmtree( "$dir/$f" );
	}
	return @rmdir( $dir );
}

function flexipipe_pct( $x ) {
	return $x === null || $x === '' ? '' : number_format( 100 * (float)$x, 1 ) . '%';
}

/** The arguments of a training run ($found: the corpus's annotations from `info training`). */
function flexipipe_train_args( $opts, $found, $stamp, $language ) {
	$a = array( 'train', '--backend', $opts['backend'], '--train-data', 'xmlfiles',
		'--ud-folder', FLEXIPIPE_SPLITS_DIR . "/$stamp",
		'--name', basename( realpath( '.' ) ) . "-$stamp",
		// absolute: some backends put a relative output folder under their own model store
		'--output-dir', realpath( FLEXIPIPE_MODELS_DIR ) . "/$stamp" );
	// the corpus's attribute names for the annotations (settings.xml: @tag for xpos; @reg / @nform / @norm …)
	foreach ( array( 'xpos' => '--xpos', 'lemma' => '--lemma', 'reg' => '--reg', 'expan' => '--expan', 'trslit' => '--trslit' ) as $k => $opt ) {
		if ( !empty( $found[ $k ]['attribute'] ) ) array_push( $a, $opt, implode( ',', $found[ $k ]['attributes'] ?? array( $found[ $k ]['attribute'] ) ) );
	}
	if ( $opts['nlpform'] === 'reg' && !empty( $found['reg'] ) ) array_push( $a, '--nlpform', 'reg' );
	if ( $opts['backend'] === 'flexitag' ) array_push( $a, '--tagpos', $opts['tagpos'] !== '' ? $opts['tagpos'] : 'auto' );
	if ( $opts['pron'] !== '' ) {
		array_push( $a, '--pron', $opts['pron'] );
		if ( $opts['pron-context'] === 'next' ) array_push( $a, '--pron-context', 'next' );
	}
	if ( $language !== '' ) array_push( $a, '--language', $language );
	return $a;
}

/** POST acts: trainstart, usemodel, mainmodel, delmodel. Returns the message to show (HTML). */
function flexipipe_train_act( $fp_settings, $fp_cmd, $jobs, $action, $act ) {
	global $user;
	$h = 'flexipipe_h';
	if ( $_SERVER['REQUEST_METHOD'] !== 'POST' ) return '';
	if ( ( $user['permissions'] ?? '' ) !== 'admin' ) return "<p class=wrong>Only administrators can train or change models.</p>";
	$models = flexipipe_corpus_models( $fp_cmd );
	$by_folder = array();
	foreach ( $models as $m ) $by_folder[ $m['folder'] ] = $m;

	if ( $act === 'trainstart' ) {
		$options = flexipipe_training_options( $fp_cmd );
		if ( !$options ) return "<p class=wrong>flexipipe could not say what this corpus can be trained on.</p>";
		$opts = array();
		foreach ( array( 'backend', 'tagpos', 'pron', 'pron-context', 'language', 'nlpform' ) as $k ) $opts[ $k ] = trim( (string)( $_POST[ $k ] ?? '' ) );
		if ( $opts['nlpform'] !== 'reg' ) $opts['nlpform'] = '';
		if ( !preg_match( '/^[a-z]{2,3}([_-][A-Za-z0-9]+)?$/', $opts['language'] ) ) $opts['language'] = '';
		$ok = false;
		foreach ( $options['backends'] as $b ) if ( $b['backend'] === $opts['backend'] && $b['can'] ) $ok = true;
		if ( !$ok ) return "<p class=wrong>That backend cannot train on this corpus.</p>";
		$avail = $options['flexitag_paths'] ?? array();
		if ( !in_array( $opts['tagpos'], $avail, true ) ) $opts['tagpos'] = count( $avail ) === 1 ? $avail[0] : '';
		if ( $opts['pron'] !== '' && !isset( $options['transcriptions'][ $opts['pron'] ] ) ) $opts['pron'] = '';
		if ( $opts['pron-context'] !== 'next' ) $opts['pron-context'] = '';
		foreach ( $jobs->all() as $j ) {
			if ( $j['kind'] === 'train' && $j['status'] === 'running' ) return "<p class=wrong>A training is still running.</p>";
		}
		// the current model stays in use until the new one is chosen
		$training = $fp_settings->training();
		$pin = ( $training['use'] ?? '' ) === '' && $models ? $models[0]['folder'] : ( $training['use'] ?? '' );
		$fp_settings->set_training( $opts + array( 'use' => $pin, 'tags' => '' ) );
		$problem = $fp_settings->save();
		if ( $problem !== '' ) return "<p class=wrong>The training settings could not be saved: " . $h( $problem ) . ".</p>";
		if ( !is_dir( FLEXIPIPE_MODELS_DIR ) && !@mkdir( FLEXIPIPE_MODELS_DIR, 0755, true ) ) return "<p class=wrong>The folder " . FLEXIPIPE_MODELS_DIR . " cannot be created.</p>";
		foreach ( (array)glob( FLEXIPIPE_SPLITS_DIR . '/*', GLOB_ONLYDIR ) as $old ) flexipipe_rmtree( $old );  // fresh splits
		// what a failed or stopped training left: folders without a model (nothing is running now)
		foreach ( (array)glob( FLEXIPIPE_MODELS_DIR . '/*', GLOB_ONLYDIR ) as $dir ) {
			if ( !file_exists( "$dir/flexipipe-model.json" ) && !file_exists( "$dir/model_vocab.json" ) ) flexipipe_rmtree( $dir );
		}
		$stamp = date( 'Ymd-His' );
		$label = "Training a $opts[backend] model on this corpus";
		$language = $opts['language'] !== '' ? $opts['language'] : $fp_settings->language();
		list( $id, $why ) = $jobs->start( 'train', $label, flexipipe_train_args( $opts, $options['found'], $stamp, $language ), $user['short'] ?? ( $user['email'] ?? '' ) );
		return $id === '' ? "<p class=wrong>The training did not start: " . $h( $why ) . ".</p>" : "<p class=right>The training has started; it runs in the background.</p>";
	}

	$folder = (string)( $_POST['folder'] ?? '' );
	if ( !isset( $by_folder[ $folder ] ) ) return "<p class=wrong>There is no such model.</p>";
	$m = $by_folder[ $folder ];
	$current = '';
	foreach ( $models as $x ) if ( !empty( $x['in_use'] ) ) $current = $x['folder'];
	$current_load = $current !== '' ? $by_folder[ $current ]['load'] : '';

	if ( $act === 'usemodel' ) {
		$fp_settings->set_training( array( 'use' => $folder ) );
		if ( $current_load !== '' && $current !== $folder ) $fp_settings->replace_model( $current_load, $m['load'] );
		$problem = $fp_settings->save();
		return $problem === '' ? "<p class=right>This corpus now uses the model trained " . $h( basename( $folder ) ) . ".</p>"
			: "<p class=wrong>The setting could not be saved: " . $h( $problem ) . ".</p>";
	}
	if ( $act === 'mainmodel' ) {
		// the corpus's model applies to its default language; a model for another language: for that one
		$fp_settings->set_training( array( 'use' => $folder ) );
		$lang = strtolower( (string)( $m['language'] ?? '' ) );
		$default = strtolower( $fp_settings->language() );
		if ( $lang !== '' && $lang !== $default ) {
			$fp_settings->set_language_model( $lang, $m['backend'], $m['load'] );
			$for = " for documents in " . $h( $lang );
		} else {
			$fp_settings->set_main_model( $m['backend'], $m['load'] );
			$for = $default !== '' ? " for documents in " . $h( $default ) . " (the corpus's language)" : '';
		}
		$problem = $fp_settings->save();
		if ( $problem !== '' ) return "<p class=wrong>The setting could not be saved: " . $h( $problem ) . ".</p>";
		$note = $lang === '' && $default === '' ? "<p class='fp-help'>The model was trained without a language and the corpus has none set: it is used where the corpus's model applies — set the corpus's language on the settings page, or retrain with a language.</p>" : '';
		return "<p class=right>The corpus now tags with the model trained " . $h( basename( $folder ) ) . " (" . $h( $m['backend'] ) . ")$for.</p>$note";
	}
	if ( $act === 'delmodel' ) {
		if ( $folder === $current ) return "<p class=wrong>The model in use cannot be removed; use another one first.</p>";
		$real = realpath( $folder );
		$base = realpath( FLEXIPIPE_MODELS_DIR );
		if ( !$real || !$base || dirname( $real ) !== $base ) return "<p class=wrong>There is no such model.</p>";
		flexipipe_rmtree( $real );
		return is_dir( $real ) ? "<p class=wrong>The model could not be removed completely.</p>" : "<p>The model was removed.</p>";
	}
	return '';
}

/** The scores of a model card in words: "upos 95.1%, lemmas 96.0%, LAS 80.2%; pronunciation 77.2% (new words 62.7%)". */
function flexipipe_model_scores( $m ) {
	$sc = $m['scores'] ?? array();
	$names = array( 'upos' => 'UPOS', 'xpos' => 'tags', 'tags' => 'tags', 'feats' => 'features', 'lemma' => 'lemmas', 'lemmas' => 'lemmas', 'las' => 'LAS' );
	$out = array();
	foreach ( $names as $k => $label ) if ( isset( $sc[ $k ] ) ) $out[] = "$label " . flexipipe_pct( $sc[ $k ] );
	$ps = $m['pron']['scores'] ?? array();
	if ( isset( $ps['all'] ) ) $out[] = 'pronunciation ' . flexipipe_pct( $ps['all'] ) . ( isset( $ps['unknown'] ) ? ' (new words ' . flexipipe_pct( $ps['unknown'] ) . ')' : '' );
	return implode( ', ', $out );
}

/** The models table. */
function flexipipe_models_table( $models, $action, $is_admin ) {
	$h = 'flexipipe_h';
	$out = "<table class='fp-table'><thead><tr><th>Trained<th>Backend<th>What it does<th>Held-out scores<th></tr></thead><tbody>";
	foreach ( $models as $m ) {
		$does = array();
		foreach ( (array)( $m['learns'] ?? array() ) as $l ) $does[] = $l;
		if ( !$does && ( $m['backend'] ?? '' ) === 'flexitag' ) $does[] = 'tags, lemmas';
		if ( !empty( $m['pron'] ) ) $does[] = 'pronunciation (@' . $h( $m['pron']['attribute'] ?? '' ) . ')' . ( !empty( $m['pron']['context'] ) ? ', with the next word' : '' );
		$folder = $h( $m['folder'] );
		$cell = !empty( $m['in_use'] ) ? '<b>in use</b> ' : '';
		if ( $is_admin ) {
			$post = function ( $act, $label, $confirm = '' ) use ( $action, $folder ) {
				return "<form method='post' action='index.php?action=$action&act=$act' style='display:inline'" . ( $confirm ? " onsubmit='return confirm(\"$confirm\")'" : '' )
					. "><input type='hidden' name='folder' value='$folder'><button type='submit'>$label</button></form> ";
			};
			if ( empty( $m['in_use'] ) ) $cell .= $post( 'usemodel', 'Use this model' );
			$cell .= $post( 'mainmodel', 'Tag with it' );
			if ( empty( $m['in_use'] ) ) $cell .= $post( 'delmodel', 'Remove', 'Remove this model?' );
			$cell .= "<a href='index.php?action=$action&act=publish&folder=" . rawurlencode( $m['folder'] ) . "'>Publish…</a>";
		}
		$out .= "<tr><td>" . $h( $m['created'] ?? '' ) . " <span class='fp-help'>" . $h( basename( $m['folder'] ) ) . "</span><td>" . $h( $m['backend'] ?? '' )
			. "<td>" . $h( implode( '; ', $does ) ) . "<td>" . $h( flexipipe_model_scores( $m ) ) . "<td>$cell</tr>";
	}
	return $out . "</tbody></table><p class='fp-help'>Scores: on documents held out from the training (each model on its own held-out documents). "
		. "The model in use gives the steps the corpus's model does not do, such as pronunciation; <i>Tag with it</i> makes it the corpus's model.</p>";
}

/** act=training (and the POST acts, which show their outcome here). */
function flexipipe_training_page( $fp_settings, $fp_cmd, $jobs, $action, $act, $is_admin ) {
	$h = 'flexipipe_h';
	@set_time_limit( 180 );  // two flexipipe calls, each starting Python
	$out = "<h1>Training models on this corpus</h1>";
	if ( $act !== 'training' ) {
		$out .= flexipipe_train_act( $fp_settings, $fp_cmd, $jobs, $action, $act );
		$fp_settings = new FlexipipeSettings();
	}

	// a training running now (followed live), the last ones
	$train_jobs = array_values( array_filter( $jobs->all(), function ( $j ) { return $j['kind'] === 'train'; } ) );
	foreach ( $train_jobs as $j ) {
		if ( $j['status'] !== 'running' ) continue;
		$out .= "<h2>" . $h( $j['label'] ) . "</h2>" . flexipipe_job_panel( $jobs, $j, $action, $is_admin );
	}
	$done = array_slice( array_values( array_filter( $train_jobs, function ( $j ) { return $j['status'] !== 'running'; } ) ), 0, 3 );
	if ( $done ) {
		$out .= "<p class='fp-help'>Last trainings: " . implode( ' · ', array_map( function ( $j ) use ( $action, $h ) {
			return "<a href='index.php?action=$action&act=job&job=" . rawurlencode( $j['id'] ) . "'>" . $h( $j['started'] ) . "</a> " . strip_tags( flexipipe_job_state( $j ) );
		}, $done ) ) . "</p>";
	}

	$models = flexipipe_corpus_models( $fp_cmd );
	$out .= "<h2>The corpus's models</h2>" . ( $models ? flexipipe_models_table( $models, $action, $is_admin ) : "<p>No model has been trained on this corpus yet.</p>" );

	$options = flexipipe_training_options( $fp_cmd );
	if ( !$options ) return $out . "<p class=wrong>flexipipe could not say what this corpus can be trained on.</p>";
	$out .= "<h2>What this corpus can teach</h2><p>In " . (int)$options['documents'] . " documents (a sample, " . number_format( (int)$options['tokens'] ) . " tokens):</p><ul>";
	foreach ( $options['found'] as $f ) $out .= "<li>" . $h( $f['label'] ) . " <span class='fp-help'>@" . $h( $f['attribute'] ) . ", " . number_format( (int)$f['tokens'] ) . " tokens</span></li>";
	$out .= "</ul>";
	if ( !$is_admin ) return $out;

	// the form
	$t = $fp_settings->training();
	$running = (bool)array_filter( $train_jobs, function ( $j ) { return $j['status'] === 'running'; } );
	$radios = '';
	$first = '';
	foreach ( $options['backends'] as $b ) {
		if ( $b['can'] && $first === '' ) $first = $b['backend'];
	}
	$want = $t['backend'] ?? $first;
	foreach ( $options['backends'] as $b ) {
		$why = $b['can'] ? 'learns ' . implode( ', ', $b['learns'] )
			: ( !$b['available'] ? 'not installed' : ( $b['lacks'] ? 'needs ' . implode( ', ', $b['lacks'] ) . ', which this corpus does not have' : 'needs ' . implode( ', ', $b['needs'] ) ) );
		$radios .= "<label" . ( $b['can'] ? '' : " class='fp-off'" ) . "><input type='radio' name='backend' value='" . $h( $b['backend'] ) . "'"
			. ( $b['can'] ? ( $want === $b['backend'] ? ' checked' : '' ) : ' disabled' ) . "> <b>" . $h( $b['backend'] ) . "</b> — " . $h( $b['about'] )
			. " <span class='fp-help'>(" . $h( $why ) . ")</span></label><br>";
	}
	$paths = array( '' => 'the best of these, tested on held-out documents', 'xpos' => 'the tags (xpos)', 'upos' => 'the UD part of speech', 'utot' => 'part of speech + features' );
	// one possible sequence: that one; several: the best of them (tested), or one chosen
	$avail = $options['flexitag_paths'] ?? array();
	$tp = "<select name='tagpos'>" . ( count( $avail ) > 1 ? "<option value=''>" . $h( $paths[''] ) . "</option>" : '' );
	foreach ( $avail as $p ) $tp .= "<option value='$p'" . ( ( $t['tagpos'] ?? '' ) === $p ? ' selected' : '' ) . ">" . $h( $paths[ $p ] ) . "</option>";
	$tp .= "</select>";
	$attrs = $fp_settings->token_attributes();
	$extra = '';
	// normalization: offered when the corpus has normalized forms (@reg, @nform, @norm …)
	if ( !empty( $options['found']['reg'] ) ) {
		$names = implode( ', ', array_map( function ( $n ) { return "@$n"; }, $options['found']['reg']['attributes'] ?? array( $options['found']['reg']['attribute'] ) ) );
		$extra .= "<tr><th>Normalization<td colspan='2'>The model learns the normalized forms (" . $h( $names ) . ") and can normalize new texts.<br>"
			. "<label><input type='checkbox' name='nlpform' value='reg'" . ( ( $t['nlpform'] ?? '' ) === 'reg' ? ' checked' : '' ) . "> Tag by the normalized forms rather than the written ones</label>"
			. " <span class='fp-help'>(historical or non-standard spelling: tags and lemmas follow the normalization — the model then only tags documents that are normalized already)</span></tr>";
	}
	foreach ( array( 'expan', 'trslit' ) as $k ) {
		if ( !empty( $options['found'][ $k ] ) ) $extra .= "<tr><th>" . $h( ucfirst( $options['found'][ $k ]['label'] ) ) . "<td colspan='2'>Learned too (@" . $h( $options['found'][ $k ]['attribute'] ) . ").</tr>";
	}
	// pronunciation: offered only for attributes that look like a transcription of the word (@phon …)
	if ( !empty( $options['transcriptions'] ) ) {
		$pr = "<select name='pron'><option value=''>no</option>";
		foreach ( $options['transcriptions'] as $k => $n ) {
			$pr .= "<option value='" . $h( $k ) . "'" . ( ( $t['pron'] ?? '' ) === $k ? ' selected' : '' ) . ">" . $h( "from " . ( $attrs[ $k ]['display'] ?? $k ) . " (@$k)" ) . "</option>";
		}
		$pr .= "</select>";
		$extra .= "<tr><th>Pronunciation<td>$pr<td class='fp-help'>Learn pronunciations from a transcription (new words by rule).</tr>"
			. "<tr><th><td><label><input type='checkbox' name='pron-context' value='next'" . ( ( $t['pron-context'] ?? '' ) === 'next' ? ' checked' : '' ) . "> It depends on the next word</label>"
			. "<td class='fp-help'>Narrow transcriptions of speech (assimilation across words); leave it off for broad ones.</tr>";
	}
	$out .= "<h2>" . ( $models ? 'Retrain' : 'Train a model' ) . "</h2>"
		. "<form method='post' action='index.php?action=$action&act=trainstart'><fieldset" . ( $running ? ' disabled' : '' ) . ">"
		. "<p>Backend:<br>$radios</p>"
		. "<table class='fp-form'>"
		. "<tr><th>flexitag tags by<td>$tp<td class='fp-help'>flexitag learns all the tags and features the corpus has, and follows one of them as the sequence it tags; the others come with each word.</tr>"
		. "<tr><th>Language<td><input name='language' size='6' value='" . $h( $t['language'] ?? $fp_settings->language() ) . "'><td class='fp-help'>The language of the documents (ISO code); the model is used for documents in it.</tr>"
		. $extra
		. "</table><p class='fp-help'>The model learns from all documents with these annotations, as they are now; some are held out to measure it. "
		. "Training runs in the background (on a large corpus it can take hours); you can leave this page. "
		. ( $models ? "The current model stays in use until you choose the new one." : "The first model is used as soon as it is ready." ) . "</p>"
		. "<p><button type='submit'>" . ( $models ? 'Retrain' : 'Train' ) . "</button>" . ( $running ? " <span class='fp-help'>(a training is running)</span>" : '' ) . "</p></fieldset></form>";
	return $out . "<p><a href='index.php?action=$action'>flexipipe settings</a></p>";
}

/** The set-up page's short version: the model in use and a link to the training page. */
function flexipipe_training_summary( $fp_cmd, $jobs, $action ) {
	$h = 'flexipipe_h';
	$models = flexipipe_corpus_models( $fp_cmd );
	$running = array_filter( $jobs->all(), function ( $j ) { return $j['kind'] === 'train' && $j['status'] === 'running'; } );
	$text = "<h2>Models trained on this corpus</h2><p>";
	if ( $running ) $text .= "A training is running. ";
	if ( $models ) {
		foreach ( $models as $m ) {
			if ( !empty( $m['in_use'] ) ) $text .= "In use: " . $h( $m['backend'] ?? '' ) . ", trained " . $h( $m['created'] ?? '' ) . ( flexipipe_model_scores( $m ) ? " (" . $h( flexipipe_model_scores( $m ) ) . ")" : '' ) . ". ";
		}
		$text .= count( $models ) . " model" . ( count( $models ) > 1 ? 's' : '' ) . " in all. ";
	} else {
		$text .= "None yet. ";
	}
	return $text . "<a href='index.php?action=$action&act=training'>Training</a></p>";
}
