<?php
/**
 * One document (action=flexipipe&cid=…): act=status shows what it contains and what a run would do
 * (`flexipipe info document`); act=process runs flexipipe on it — keeping existing annotation unless
 * the user ticks "Replace" for it (per attribute), never when the file cannot be written — and shows the
 * outcome (no immediate redirect: errors stay visible). flexipipe backs the file up before it
 * writes (backups/, TEITOK's backups page lists it) and refuses a result that is not valid XML or
 * would change the text. Included by flexipipe.php ($fp_settings, $fp_cmd).
 *
 * The run is synchronous for now (one document, the time limit lifted); the background job with
 * progress is the next step (dev/TEITOK-DOCUMENT-PIPELINE.md, phase 2).
 */

function flexipipe_document_status( $fp_cmd, $path, $manifest, $tasks, $keep = 'all', $choice = array() ) {
	$a = array( 'info', 'document', '--input', $path, '--output-format', 'json', '--keep', $keep );
	if ( $tasks !== '' ) array_push( $a, '--tasks', $tasks );
	if ( !empty( $choice['backend'] ) ) {  // chosen for this run
		array_push( $a, '--backend', $choice['backend'], '--model', $choice['model'] ?? '' );
	} elseif ( !empty( $manifest['backend'] ) ) {
		array_push( $a, '--backend', $manifest['backend'] );
		if ( !empty( $manifest['model'] ) ) array_push( $a, '--model', $manifest['model'] );
	}
	if ( !empty( $choice['language'] ) ) array_push( $a, '--language', $choice['language'] );
	if ( !empty( $choice['layout'] ) ) array_push( $a, '--sentence-layout', $choice['layout'] );
	list( $rc, $out, $err ) = $fp_cmd->run( $a );
	$st = json_decode( trim( $out ), true );
	if ( !is_array( $st ) || isset( $st['error'] ) ) {
		$why = is_array( $st ) && isset( $st['error'] ) ? $st['error'] : trim( preg_replace( '/^.*\n/s', '', trim( $err ) ) );
		return array( null, $why );
	}
	return array( $st, '' );
}

/** The status as HTML: contents, history, the planned run and what it would change. */
function flexipipe_document_status_html( $st, $is_admin, $replace = array(), $action = 'flexipipe', $cid = '', $controls = array() ) {
	$h = 'flexipipe_h';
	$run = $st['run'];
	$out = "<h2>This document</h2><table class='fp-table'>"
		. "<tr><th>Tokens<td>" . (int)$st['tokens'] . "</tr>"
		. "<tr><th>Sentences<td>" . (int)$st['sentences'] . "</tr>"
		. ( $st['entities'] ? "<tr><th>Named entities<td>" . (int)$st['entities'] . "</tr>" : '' );
	foreach ( $st['attributes'] as $name => $a ) {
		$out .= "<tr><th>" . $h( ucfirst( $name ) ) . "<td>" . (int)$a['tokens'] . " tokens <span class='fp-help'>@" . $h( $a['attribute'] ) . "</span></tr>";
	}
	$out .= "</table>";
	if ( $st['changes'] ) {
		$out .= "<details><summary>Earlier changes (" . count( $st['changes'] ) . ")</summary><ul>";
		foreach ( array_reverse( $st['changes'] ) as $c ) {
			$out .= "<li>" . $h( trim( ( $c['when'] ?? '' ) . ' ' . ( $c['who'] ?? '' ) ) ) . ": " . $h( $c['text'] ?? ( $c['tasks'] ?? '' ) ) . "</li>";
		}
		$out .= "</ul></details>";
	}

	$lang_from = array( 'document' => "from the document", 'corpus' => "the corpus's default" );
	$lang_from['given'] = 'chosen for this run';
	$model_from = array( 'corpus' => "the corpus's setting", 'flexipipe' => "flexipipe's choice", 'given' => "chosen for this run" );
	$lang_now = $run['language'] ? $h( $run['language'] ) . " <span class='fp-help'>" . $h( $lang_from[ $run['language_source'] ] ?? '' ) . "</span>" : "detected from the text when it runs";
	$model_now = $run['backend'] ? $h( $run['backend'] . ' / ' . $run['model'] ) . " <span class='fp-help'>" . $h( $model_from[ $run['model_source'] ] ?? '' ) . "</span>" : "chosen when it runs";
	$corpus = $is_admin ? " <a class='fp-help' href='index.php?action=$action'>change for the corpus</a>" : '';
	$out .= "<h2>A flexipipe run</h2><table class='fp-table'>"
		. "<tr><th>Language<td>" . ( $controls['language'] ?? $lang_now ) . $corpus . "</tr>"
		. "<tr><th>Model<td>" . ( $controls['model'] ?? $model_now ) . $corpus . "</tr>"
		. ( isset( $controls['layout'] ) ? "<tr><th>Sentences<td>" . $controls['layout'] . $corpus . "</tr>" : '' )
		. "<tr><th>Steps<td>" . flexipipe_document_steps_html( $run, $is_admin, $action ) . "</tr></table>";

	$changes = array();
	foreach ( $run['kept'] ?? array() as $o ) {
		if ( in_array( $o['name'], $replace, true ) ) $changes[] = "<b>replace</b> the existing " . $h( $o['name'] ) . " of " . (int)$o['tokens'] . " tokens (@" . $h( $o['attribute'] ) . ")";
		else $changes[] = "keep the existing " . $h( $o['name'] ) . " of " . (int)$o['tokens'] . " tokens as it is (@" . $h( $o['attribute'] ) . ")";
	}
	if ( $run['new_attributes'] ) {
		$changes[] = "add " . implode( ', ', array_map( function ( $a ) use ( $h ) { return $h( $a['name'] ) . " (@" . $h( $a['attribute'] ) . ")"; }, $run['new_attributes'] ) );
	}
	if ( in_array( 'sentences', $run['adds'], true ) ) {
		$lay = $run['sentence_layout'] ?? array();
		$changes[] = "add sentences (&lt;s&gt;), placed as for " . $h( $lay['preset'] ?? 'written' ) . " text"
			. " <span class='fp-help'>(" . $h( implode( '; ', array_map( function ( $n, $r ) { return "$n: {$r['boundary']}, {$r['crossing']}"; },
				array_keys( $lay['roles'] ?? array() ), array_values( $lay['roles'] ?? array() ) ) ) ) . ")</span>"
			. " — <button type='submit' form='fp-run' formaction='" . $h( "index.php?action=$action&act=preview&cid=" . rawurlencode( $cid ) ) . "'>Preview the sentences</button>";
	}
	if ( in_array( 'tokens', $run['adds'], true ) ) $changes[] = "split the text into tokens";
	if ( $changes ) {
		$out .= "<p>The run would:</p><ul><li>" . implode( "</li><li>", $changes ) . "</li></ul>"
			. ( $is_admin ? "<p class='fp-help'><a href='index.php?action=$action#fp-layers'>Where flexipipe writes</a> (which attribute each layer goes to) is set for the corpus.</p>" : '' );
	}
	return $out;
}

/** The steps of the run, as users know them (also the run form's checkboxes). */
function flexipipe_step_labels() {
	return array( 'tokenize' => 'tokenize', 'segment' => 'sentences', 'normalize' => 'normalize', 'lemmatize' => 'lemmatize',
		'tag' => 'tag', 'parse' => 'parse', 'ner' => 'named entities', 'pron' => 'pronunciation (G2P)' );
}

/** The run's steps, and who does those the model does not do ("named entities: nametag / … (the corpus's setting)"). */
function flexipipe_document_steps_html( $run, $is_admin, $action ) {
	$h = 'flexipipe_h';
	$labels = flexipipe_step_labels();
	$from = array( 'corpus' => "the corpus's setting", 'given' => 'chosen for this run', 'default' => 'installed for this language' );
	$main = array();
	$other = array();
	foreach ( $run['steps'] ?? array() as $st ) {
		$label = $labels[ $st['task'] ] ?? $st['task'];
		if ( empty( $st['backend'] ) ) {
			$other[] = $h( $label ) . ": <span class=wrong>not done by this model</span>"
				. ( $is_admin ? " <a class='fp-help' href='index.php?action=$action'>choose a backend for it</a>" : " <span class='fp-help'>(left out)</span>" );
		} elseif ( in_array( $st['source'], array( 'main', 'flexipipe' ), true ) ) {
			$main[] = $h( $label );
		} else {
			$other[] = $h( $label ) . ": " . flexipipe_step_who( $st, $is_admin ) . " <span class='fp-help'>(" . $h( $from[ $st['source'] ] ?? '' ) . ")</span>";
		}
	}
	if ( !isset( $run['steps'] ) ) $main = array_map( $h, $run['tasks'] );  // an older flexipipe: no plan per step
	return implode( ', ', $main ) . ( $main && $other ? "<br>" : '' ) . implode( "<br>", $other );
}

/** Who does a step, in words: a model on disk by what it is ("a locally trained model: trained on
 *  this corpus (ortofon), 2026-10-18; …"), other models by backend / name. */
function flexipipe_step_who( $st, $is_admin ) {
	$h = 'flexipipe_h';
	if ( !empty( $st['about'] ) ) {
		$card = $st['card'] ?? array();
		$what = !empty( $card['locally_trained'] ) ? 'a model trained locally' : 'an installed model';
		$tip = $is_admin ? " title='" . $h( ( $card['name'] ?? '' ) . ' — ' . ( $card['path'] ?? '' ) ) . "'" : '';
		// "locally trained on this corpus (ortofon), …": the "locally trained" is said once, in front
		$about = preg_replace( '/^locally trained /', '', $st['about'] );
		return "<span$tip>" . $h( !empty( $card['locally_trained'] ) ? "$what $about" : "$what: $about" ) . "</span>";
	}
	return $h( $st['backend'] . ( $st['model'] ? ' / ' . $st['model'] : '' ) );
}

function flexipipe_document_page( $fp_settings, $fp_cmd, $action, $act, $cid ) {
	global $user;
	$is_admin = ( $user['permissions'] ?? '' ) === 'admin';
	$h = 'flexipipe_h';
	@set_time_limit( 180 );  // several flexipipe calls, each starting Python (a run lifts the limit itself)
	$out = "<h1>flexipipe</h1>";
	$path = FlexipipeCmd::document_path( $cid );
	if ( $path === '' ) return $out . "<p class=wrong>There is no document " . $h( $cid ) . " in this corpus.</p>";
	$doclink = "index.php?action=file&cid=" . rawurlencode( $cid );
	$out .= "<p>Document: <a href='$doclink'>" . $h( $cid ) . "</a></p>";

	list( $mkey, $manifest ) = FlexipipeCmd::manifest_for( $fp_settings, $path, (string)( $_GET['manifest'] ?? '' ) );
	$tasks = isset( $_POST['tasks'] ) ? implode( ',', array_map( 'strval', (array)$_POST['tasks'] ) ) : (string)( $manifest['tasks'] ?? '' );
	// chosen for this run (else: the document's / the corpus's settings)
	$choice = array( 'language' => trim( (string)( $_POST['fp_lang'] ?? '' ) ), 'layout' => trim( (string)( $_POST['fp_layout'] ?? '' ) ) );
	$bm = trim( (string)( $_POST['fp_model'] ?? '' ) );
	if ( strpos( $bm, ':' ) > 0 ) list( $choice['backend'], $choice['model'] ) = explode( ':', $bm, 2 );
	if ( !isset( flexipipe_layouts()[ $choice['layout'] ] ) ) $choice['layout'] = '';
	list( $st, $problem ) = flexipipe_document_status( $fp_cmd, $path, $manifest, $tasks, 'all', $choice );
	if ( !$st ) return $out . "<p class=wrong>flexipipe could not read this document.</p>" . ( $is_admin && $problem ? "<p class='fp-help'>" . $h( $problem ) . "</p>" : '' );

	$writable = !empty( $st['writable'] );
	// existing annotation is kept; it is only replaced where the user ticks "Replace" (per attribute)
	$kept_names = array_map( function ( $o ) { return $o['name']; }, $st['run']['kept'] ?? array() );
	$replace = array_values( array_intersect( array_map( 'strval', (array)( $_POST['replace'] ?? array() ) ), $kept_names ) );
	$keep = implode( ',', array_diff( $kept_names, $replace ) );

	if ( $act === 'preview' ) {
		$out .= flexipipe_document_preview( $fp_cmd, $path, $manifest, $st['run'] + array( 'layout' => $choice['layout'] ) );
	}
	if ( $act === 'process' && $_SERVER['REQUEST_METHOD'] === 'POST' ) {
		if ( !$writable ) {
			$out .= "<p class=wrong>flexipipe cannot save this document: the web server may not change it or its folder.</p>";
		} else {
			return $out . flexipipe_document_run( $fp_cmd, $path, $manifest, $tasks, $keep, implode( ',', $replace ), $st['run'] + array( 'layout' => $choice['layout'] ), $doclink, $is_admin, $action, $cid );
		}
	}

	$out .= flexipipe_document_status_html( $st, $is_admin, $replace, $action, $cid, flexipipe_document_controls( $fp_cmd, $st['run'], $choice ) );
	if ( !$writable ) {
		$who = function_exists( 'posix_geteuid' ) ? ( posix_getpwuid( posix_geteuid() )['name'] ?? '' ) : '';
		$out .= "<p class=wrong>flexipipe cannot save this document: the web server may not change the file or its folder.</p>"
			. ( $is_admin ? "<p class='fp-help'>" . $h( ( $who ? "$who needs" : "The web server needs" ) . " write permission on $path or xmlfiles/." ) . "</p>" : '' );
		return $out;
	}
	$checks = '';
	$providers = $st['run']['providers'] ?? array();
	foreach ( flexipipe_step_labels() as $t => $label ) {
		// a step nobody does (not the model, no backend chosen for it) is not offered
		if ( array_key_exists( $t, $providers ) && $providers[ $t ] === null ) {
			$checks .= "<label class='fp-off' title='The model does not do this step, and no backend is chosen for it in the corpus settings'><input type='checkbox' disabled> " . $h( $label ) . "</label> ";
			continue;
		}
		$on = in_array( $t, $st['run']['tasks'], true ) ? ' checked' : '';
		$by = $providers[ $t ] ?? null;
		$tip = $by && !in_array( $by['source'] ?? '', array( 'main', 'flexipipe' ), true )
			? " title='" . $h( strip_tags( flexipipe_step_who( $by, false ) ) ) . "'" : '';
		$checks .= "<label$tip><input type='checkbox' name='tasks[]' value='$t'$on> " . $h( $label ) . "</label> ";
	}
	$out .= "<form method='post' id='fp-run' action='index.php?action=$action&act=process&cid=" . rawurlencode( $cid ) . "'>
		<p>Steps: $checks <button type='submit' formaction='index.php?action=$action&act=status&cid=" . rawurlencode( $cid ) . "'>Show what these steps would do</button></p>";
	foreach ( $st['run']['kept'] ?? array() as $o ) {
		$on = in_array( $o['name'], $replace, true ) ? ' checked' : '';
		$out .= "<p><label><input type='checkbox' name='replace[]' value='" . $h( $o['name'] ) . "'$on> Replace the existing "
			. $h( $o['name'] ) . " of " . (int)$o['tokens'] . " tokens <span class='fp-help'>(@" . $h( $o['attribute'] ) . "; otherwise kept, and used as given where the tagger can)</span></label></p>";
	}
	$out .= "<p><button type='submit'>Run flexipipe on this document</button>
		<span class='fp-help'>A copy of the document as it is now is kept in the backups (file management → backups).</span></p></form>";
	return $out;
}

/** Run flexipipe (synchronous) and show the outcome. */
function flexipipe_document_run( $fp_cmd, $path, $manifest, $tasks, $keep, $replace, $plan, $doclink, $is_admin, $action, $cid ) {
	$h = 'flexipipe_h';
	@set_time_limit( 0 );
	$started = microtime( true );
	// what the status page showed is what runs: its language, backend and model
	$use = is_array( $manifest ) ? $manifest : array();
	if ( !empty( $plan['backend'] ) && ( empty( $use['backend'] ) || ( $plan['model_source'] ?? '' ) === 'given' ) ) {
		$use['backend'] = $plan['backend'];
		$use['model'] = $plan['model'] ?? '';
	}
	$opt = array( 'tasks' => $tasks, 'language' => $plan['language'] ?? '', 'keep' => $keep, 'replace' => $replace, 'layout' => $plan['layout'] ?? '' );
	list( $rc, $out, $err ) = $fp_cmd->run( FlexipipeCmd::process_args( $path, $use, $opt ) );
	$secs = round( microtime( true ) - $started, 1 );
	$summary = null;
	foreach ( array_reverse( explode( "\n", trim( $out ) ) ) as $line ) {
		$j = json_decode( $line, true );
		if ( is_array( $j ) && isset( $j['status'] ) ) { $summary = $j; break; }
	}
	// flexipipe's own explanation of a refusal or failure (the [flexipipe] lines on stderr)
	$reasons = array();
	foreach ( explode( "\n", $err ) as $line ) {
		if ( preg_match( '/^\[flexipipe\].*(refus|failed|error|cannot|not well-formed|would change)/i', $line ) ) $reasons[] = trim( $line );
	}
	if ( !$reasons ) {  // none of the known refusals: flexipipe's last message(s)
		$said = array_values( array_filter( array_map( 'trim', explode( "\n", $err ) ), function ( $l ) { return strpos( $l, '[flexipipe]' ) === 0 || preg_match( '/Error|error:/', $l ); } ) );
		$reasons = array_slice( $said, -2 );
	}
	$ok = $rc === 0 && ( !$summary || $summary['status'] === 'ok' );
	if ( $ok ) {
		$res = "<p class=right>flexipipe annotated the document ($secs s)"
			. ( $summary && !empty( $summary['backend'] ) ? " with " . $h( $summary['backend'] . ' / ' . ( $summary['model'] ?? '' ) ) : '' ) . ".</p>"
			. "<p><a href='$doclink'>View the document</a> · <a href='index.php?action=$action&act=status&cid=" . rawurlencode( $cid ) . "'>What it contains now</a></p>";
	} else {
		$res = "<p class=wrong>flexipipe did not change the document.</p>";
		if ( $reasons ) $res .= "<ul><li>" . implode( "</li><li>", array_map( $h, array_slice( $reasons, -3 ) ) ) . "</li></ul>";
		$res .= "<p><a href='$doclink'>Back to the document</a></p>";
	}
	if ( $is_admin ) {
		$tail = implode( "\n", array_slice( explode( "\n", trim( $err ) ), -25 ) );
		$res .= "<details><summary>flexipipe's output</summary><pre>" . $h( "exit $rc\n" . trim( $out ) . "\n\n" . $tail ) . "</pre></details>";
	}
	return $res;
}


/** A trial run of the sentence step only (--test: nothing is written): how the sentences would be placed. */
function flexipipe_document_preview( $fp_cmd, $path, $manifest, $plan ) {
	$h = 'flexipipe_h';
	@set_time_limit( 0 );
	$use = is_array( $manifest ) ? $manifest : array();
	if ( !empty( $plan['backend'] ) && ( empty( $use['backend'] ) || ( $plan['model_source'] ?? '' ) === 'given' ) ) {
		$use['backend'] = $plan['backend'];
		$use['model'] = $plan['model'] ?? '';
	}
	$args = FlexipipeCmd::process_args( $path, $use, array( 'tasks' => 'segment', 'language' => $plan['language'] ?? '', 'keep' => 'all', 'test' => 1, 'verbose' => 1, 'layout' => $plan['layout'] ?? '' ) );
	list( $rc, $out, $err ) = $fp_cmd->run( $args );
	if ( preg_match( '/\[flexipipe\] sentences \(([^)]*)\): (\d+) placed, (\d+) split into (\d+) parts, (\d+) as token lists, (\d+) cut at hard edges(?:; elements split: ([^\n]*))?/', $err, $m ) ) {
		$res = "<div class='fp-preview'><b>Preview</b> (nothing written): " . (int)$m[2] . " sentences";
		$bits = array();
		if ( (int)$m[3] ) $bits[] = (int)$m[3] . " of them split into " . (int)$m[4] . " linked parts around elements that stay whole";
		if ( (int)$m[5] ) $bits[] = (int)$m[5] . " as token lists";
		if ( (int)$m[6] ) $bits[] = (int)$m[6] . " ends added at " . $h( 'hard edges' ) . " (e.g. speaker turns)";
		if ( !empty( $m[7] ) ) $bits[] = "elements split: " . $h( $m[7] );
		return $res . ( $bits ? ": " . implode( '; ', $bits ) : '' ) . ".</div>";
	}
	if ( $rc === 0 ) return "<div class='fp-preview'><b>Preview</b>: this document would get no new sentences (it has them already, or the step does not run).</div>";
	return "<p class=wrong>The preview did not work.</p><pre class='fp-help'>" . $h( implode( "\n", array_slice( explode( "\n", trim( $err ) ), -5 ) ) ) . "</pre>";
}


/** The run's choices as fields of the run form (#fp-run): language, model, sentence layout. */
function flexipipe_document_controls( $fp_cmd, $run, $choice ) {
	$h = 'flexipipe_h';
	$lang = $run['language'] ?? '';
	$langs = $fp_cmd->json( array( 'info', 'languages', '--output-format', 'json' ) );
	$l = "<select name='fp_lang' form='fp-run' onchange='this.form.querySelector(\"[name=fp_model]\") && (this.form.elements.fp_model.value=\"\")'>"
		. "<option value=''>" . ( $lang !== '' && $choice['language'] === '' ? $h( "$lang (" . ( $run['language_source'] === 'document' ? "from the document" : "the corpus's default" ) . ")" ) : "the document's or the corpus's" ) . "</option>";
	$names = array();
	foreach ( $langs['languages'] ?? array() as $e ) {
		$code = $e['iso_639_1'] ?? ( $e['iso_639_3'] ?? '' );
		if ( $code !== '' ) $names[ $code ] = $e['name'] ?? $code;
	}
	asort( $names );
	foreach ( $names as $code => $name ) {
		$l .= "<option value='" . $h( $code ) . "'" . ( $choice['language'] === $code ? ' selected' : '' ) . ">" . $h( "$name ($code)" ) . "</option>";
	}
	$l .= "</select>";

	$m = "<select name='fp_model' form='fp-run'><option value=''>";
	$src = array( 'corpus' => "the corpus's setting", 'flexipipe' => "flexipipe's choice" );
	$m .= $run['backend'] && ( $run['model_source'] ?? '' ) !== 'given'
		? $h( $run['backend'] . ' / ' . $run['model'] . ' (' . ( $src[ $run['model_source'] ] ?? '' ) . ')' )
		: "the corpus's setting or flexipipe's choice";
	$m .= "</option>";
	if ( $lang !== '' ) {
		$data = $fp_cmd->json( array( 'info', 'models', '--language', $lang, '--output-format', 'json' ) );
		$chosen = !empty( $choice['backend'] ) ? $choice['backend'] . ':' . $choice['model'] : '';
		foreach ( $data['models'] ?? array() as $e ) {
			if ( empty( $e['backend'] ) || empty( $e['model'] ) ) continue;
			$v = $e['backend'] . ':' . $e['model'];
			$m .= "<option value='" . $h( $v ) . "'" . ( $v === $chosen ? ' selected' : '' ) . ">" . $h( $e['backend'] . ' / ' . $e['model'] ) . ( ( $e['installed'] ?? null ) === false ? ' (not downloaded)' : '' ) . "</option>";
		}
	}
	$m .= "</select>";

	$corpus_layout = $run['sentence_layout']['corpus_preset'] ?? 'written';
	$y = "<select name='fp_layout' form='fp-run'><option value=''>" . $h( flexipipe_layouts()[ $corpus_layout ] ?? $corpus_layout ) . " (the corpus's)</option>";
	foreach ( flexipipe_layouts() as $k => $label ) {
		if ( $k === $corpus_layout ) continue;
		$y .= "<option value='" . $h( $k ) . "'" . ( $choice['layout'] === $k ? ' selected' : '' ) . ">" . $h( $label ) . "</option>";
	}
	$y .= "</select>";
	return array( 'language' => $l, 'model' => $m, 'layout' => $y );
}
