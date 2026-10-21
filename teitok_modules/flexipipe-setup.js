/**
 * flexipipe set-up page (flexipipe_setup.php): the language and model lists of the defaults form,
 * filled from ?api=languages and ?api=models (cached server-side), and the rows of models per
 * language. Data from the page: window.flexipipeSetup = {default_lang, default_backend,
 * default_model, per_lang: [{code, backend, model}], api}.
 */
(function () {
	'use strict';
	var cfg = window.flexipipeSetup || { per_lang: [] };
	var languages = null;          // [{code, name}]
	var modelsByLang = {};         // code -> Promise<[{backend, model, installed}]>

	function api(name, params) {
		var url = new URL(cfg.api || window.location.href, window.location.href);
		url.searchParams.set('api', name);
		Object.keys(params || {}).forEach(function (k) { url.searchParams.set(k, params[k]); });
		return fetch(url.toString(), { credentials: 'same-origin' }).then(function (r) { return r.json(); });
	}

	function loadLanguages() {
		if (!languages) languages = api('languages').then(function (d) { return (d && d.languages) || []; }).catch(function () { return []; });
		return languages;
	}

	function loadModels(code) {
		if (!code) return Promise.resolve([]);
		if (!modelsByLang[code]) modelsByLang[code] = api('models', { language: code }).then(function (d) {
			var list = (d && d.models) || [];
			list.auto = d && d.auto;  // {backend, model}: flexipipe's current choice
			return list;
		}).catch(function () { return []; });
		return modelsByLang[code];
	}

	function option(value, label, selected) {
		var o = document.createElement('option');
		o.value = value;
		o.textContent = label;
		if (selected) o.selected = true;
		return o;
	}

	/** While a list loads: one disabled entry saying so. */
	function loading(select, text) {
		select.innerHTML = '';
		select.appendChild(option('', text, true));
		select.disabled = true;
	}

	function fillLanguages(select, chosen, firstLabel) {
		loading(select, 'Loading languages…');
		return loadLanguages().then(function (list) {
			select.disabled = false;
			select.innerHTML = '';
			select.appendChild(option('', firstLabel, !chosen));
			var known = false;
			list.forEach(function (l) {
				if (l.code === chosen) known = true;
				select.appendChild(option(l.code, l.name + ' (' + l.code + ')', l.code === chosen));
			});
			if (chosen && !known) select.appendChild(option(chosen, chosen, true));  // keep a code flexipipe does not list
		});
	}

	/** The model select for a language; its value "backend:model" goes to the two hidden inputs. */
	function fillModels(select, code, backend, model, beIn, moIn) {
		var want = backend && model ? backend + ':' + model : '';
		if (code) loading(select, 'Loading models…');
		return loadModels(code).then(function (list) {
			select.disabled = false;
			select.innerHTML = '';
			var auto = list.auto && list.auto.backend ? ' (now: ' + list.auto.backend + ' / ' + list.auto.model + ')' : '';
			select.appendChild(option('', "flexipipe's choice" + auto, !want));
			var known = false;
			list.forEach(function (m) {
				var v = m.backend + ':' + m.model;
				if (v === want) known = true;
				select.appendChild(option(v, (m.label || m.backend + ' / ' + m.model) + (m.installed === false ? ' (not downloaded)' : ''), v === want));
			});
			if (want && !known) select.appendChild(option(want, backend + ' / ' + model, true));
			sync(select, beIn, moIn);
		});
	}

	function sync(select, beIn, moIn) {
		var v = select.value || '';
		var i = v.indexOf(':');
		beIn.value = i > 0 ? v.slice(0, i) : '';
		moIn.value = i > 0 ? v.slice(i + 1) : '';
	}

	function addLanguageRow(entry) {
		var tbody = document.querySelector('#fp-langs tbody');
		var tr = document.createElement('tr');
		tr.innerHTML = "<td><select name='lang_code[]'></select><td><select></select>" +
			"<input type='hidden' name='lang_backend[]'><input type='hidden' name='lang_model[]'>" +
			"<td><button type='button' title='Remove'>×</button>";
		tbody.appendChild(tr);
		var langSel = tr.querySelector("select[name='lang_code[]']");
		var modelSel = tr.querySelectorAll('select')[1];
		var beIn = tr.querySelector("input[name='lang_backend[]']");
		var moIn = tr.querySelector("input[name='lang_model[]']");
		fillLanguages(langSel, entry.code || '', '—').then(function () {
			return fillModels(modelSel, langSel.value, entry.backend, entry.model, beIn, moIn);
		});
		langSel.addEventListener('change', function () { fillModels(modelSel, langSel.value, '', '', beIn, moIn); });
		modelSel.addEventListener('change', function () { sync(modelSel, beIn, moIn); });
		tr.querySelector('button').addEventListener('click', function () { tr.remove(); });
	}

	document.addEventListener('DOMContentLoaded', function () {
		var langSel = document.getElementById('fp-lang');
		var modelSel = document.getElementById('fp-model');
		var beIn = document.getElementById('fp-backend-in');
		var moIn = document.getElementById('fp-model-in');
		if (!langSel || !modelSel) return;
		beIn.value = cfg.default_backend || '';
		moIn.value = cfg.default_model || '';
		fillLanguages(langSel, cfg.default_lang || '', 'Detect for each document').then(function () {
			return fillModels(modelSel, langSel.value, cfg.default_backend, cfg.default_model, beIn, moIn);
		});
		langSel.addEventListener('change', function () { fillModels(modelSel, langSel.value, '', '', beIn, moIn); });
		modelSel.addEventListener('change', function () { sync(modelSel, beIn, moIn); });
		(cfg.per_lang || []).forEach(addLanguageRow);
		document.getElementById('fp-add-lang').addEventListener('click', function () { addLanguageRow({}); });
	});
})();
